use super::evidence::{Date, session_date};
use crate::{Error, Result};
use serde_json::{Value, json};
use std::{collections::BTreeSet, net::SocketAddr, time::Duration};

const INSTRUCTION: &str = concat!(
    "你负责记忆检索。",
    "只分析问题，不作答，检索用语保持与原问题同一语言。",
    "intent 选 lookup（普通事实、时间、计算和比较）、recommendation（用户要适合自己情况的建议、选择或方案）、inventory（需要清点、列举不同物件或事件，包括数有几件或几次）。",
    "比较或计算若依赖分别存储的事实，把每个必要操作数写成独立子问题，保留对象与时间；",
    "并列问题也拆开。",
    "单一事实和清点目标不强行拆分。",
    "subquestions 为零或二至四条，不补充事实。",
);
const FACT_INSTRUCTION: &str = concat!(
    "逐段核对清点题，只提取原文明确满足提问条件的物品或事件。",
    "无关的段落返回空数组。",
    "每条先逐字摘录包括动作、时态和否定词的完整证据句 quote，再逐字摘录其中指向物品或事件的短语 mention。",
    "按证据判断 status：confirmed 表示原文证实它满足问题的条件；",
    "不符合所问状态的为 excluded，例如问已发生时排除仅有计划的事。",
    "label 写能区分实例的具体名称或场合。",
    "event_day 仅用于清点一次性活动，逐字抄录证据句中指明活动发生日的词（如今天、昨天、绝对日期）。",
    "清点物品、跨日事件、日期缺失或含糊时填空字符串。",
    "不要补全原文缺失的信息。",
);
const SCOPE_INSTRUCTION: &str = concat!(
    "逐条审查清点清单中的一个条目。",
    "一条原文只要贡献一个符合条件的例子就选入（accept），不要求列出所有答案。",
    "先摘出原文对该条目的动作或状态，再判断条目是否属于所问类别、是否符合所问行为。",
    "不同类别、状态不符或无证据时为 reject。",
    "reason 限一句。",
);
const MATCH_INSTRUCTION: &str = concat!(
    "你在合并用户日记的清点记录。",
    "比较两段中给出的具体条目。",
    "先对比物品种类或事件、地点和能区分实例的线索，用一句话说明。",
    "物品种类不同一定是 different。",
    "同一物品的购买、送到、保养是它的不同阶段；",
    "‘那张’等回指与名称、位置吻合时判 same。",
    "两段信息详略不同本身不是矛盾。",
    "对于活动，用会话日期解释相对时间，同一次事件的回顾为 same，不同日期又做一次为 different。",
    "有证据为同一实例才合并。",
);

pub(super) struct InventoryFact {
    label: String,
    quote: String,
    session: String,
    event_day: Option<Date>,
    identity: String,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum QueryIntent {
    #[default]
    Lookup,
    Recommendation,
    Inventory,
}

pub(super) struct Plan {
    pub intent: QueryIntent,
    pub variants: Vec<String>,
}

#[derive(Clone)]
pub(super) struct Splitter {
    agent: ureq::Agent,
    url: String,
}

impl Splitter {
    pub(super) fn from_env() -> Result<Self> {
        let address = match std::env::var("MEMORYH_SPLITTER_ADDR") {
            Ok(address) => address.parse().map_err(split_error)?,
            Err(std::env::VarError::NotPresent) => SocketAddr::from(([127, 0, 0, 1], 8080)),
            Err(error) => return Err(split_error(error)),
        };
        Self::new(address)
    }

    pub(super) fn new(addr: SocketAddr) -> Result<Self> {
        if !addr.ip().is_loopback() {
            return Err(Error::InvalidQuery(
                "splitter requires a local loopback socket".into(),
            ));
        }
        let agent = ureq::Agent::config_builder()
            .proxy(None)
            .max_redirects(0)
            .timeout_global(Some(Duration::from_secs(120)))
            .build()
            .into();
        Ok(Self {
            agent,
            url: format!("http://{addr}/v1/chat/completions"),
        })
    }
    pub(super) fn plan(&self, query: &str) -> Result<Plan> {
        let schema = json!({"type":"object","properties":{
            "intent":{"type":"string","enum":["lookup","recommendation","inventory"]},
            "subquestions":{"type":"array","items":{"type":"string"},"maxItems":4}
        },"required":["intent","subquestions"],"additionalProperties":false});
        let output = self.complete(INSTRUCTION, query, schema, 384)?;
        parse(query, &output)
    }

    pub(super) fn facts(
        &self,
        query: &str,
        body: &str,
        date_header: &str,
        session_day: Option<Date>,
        known: &mut Vec<InventoryFact>,
    ) -> Result<BTreeSet<String>> {
        let schema = json!({"type":"object","properties":{"facts":{"type":"array","items":{
            "type":"object","properties":{"quote":{"type":"string"},"mention":{"type":"string"},
            "status":{"type":"string","enum":["confirmed","excluded"]},"label":{"type":"string"},"event_day":{"type":"string"}},
            "required":["quote","mention","status","label","event_day"],"additionalProperties":false}}},"required":["facts"],"additionalProperties":false});
        let mut found = BTreeSet::new();
        for part in chunks(body) {
            let data = json!({"question":query, "session":date_header, "text":part});
            let output = self.complete(FACT_INSTRUCTION, &data.to_string(), schema.clone(), 640)?;
            let items = output["facts"]
                .as_array()
                .ok_or_else(|| split_error("missing facts"))?;
            for item in items {
                let Some((label, quote)) = grounded_fact(item, part) else {
                    continue;
                };
                let candidate = json!({"问题":query, "候选条目":label, "证据句":quote, "会话":date_header, "原文":part});
                if !self.verdict(SCOPE_INSTRUCTION, &candidate, "reject", "accept")? {
                    continue;
                }
                let day = item["event_day"]
                    .as_str()
                    .and_then(|span| occurrence_day(span, quote, session_day));
                let fact = InventoryFact {
                    identity: label.into(),
                    label: label.into(),
                    quote: quote.into(),
                    session: date_header.into(),
                    event_day: day,
                };
                let key = self.identify(query, fact, known)?;
                found.insert(key);
            }
        }
        Ok(found)
    }

    fn identify(
        &self,
        query: &str,
        mut fact: InventoryFact,
        known: &mut Vec<InventoryFact>,
    ) -> Result<String> {
        for previous in known.iter() {
            if matches!((fact.event_day, previous.event_day), (Some(a), Some(b)) if a != b) {
                continue;
            }
            let input = json!({"问题":query, "新条目":{"名称":fact.label,"证据":fact.quote,"会话":fact.session,"发生日":fact.event_day.map(|d| d.to_string())},
                "已有条目":{"名称":previous.label,"证据":previous.quote,"会话":previous.session,"发生日":previous.event_day.map(|d| d.to_string())}});
            if self.verdict(MATCH_INSTRUCTION, &input, "different", "same")? {
                return Ok(previous.identity.clone());
            }
        }
        let mut suffix = 2;
        while known.iter().any(|f| f.identity == fact.identity) {
            fact.identity = format!("{} #{suffix}", fact.label);
            suffix += 1;
        }
        let identity = fact.identity.clone();
        known.push(fact);
        Ok(identity)
    }

    fn verdict(&self, instruction: &str, input: &Value, no: &str, yes: &str) -> Result<bool> {
        let schema = json!({"type":"object","properties":{"reason":{"type":"string"},
            "verdict":{"type":"string","enum":[no,yes]}},
            "required":["reason","verdict"],"additionalProperties":false});
        let result = self.complete(instruction, &input.to_string(), schema, 256)?;
        match result["verdict"].as_str() {
            Some(v) if v == yes => Ok(true),
            Some(v) if v == no => Ok(false),
            _ => Err(split_error("invalid evidence verdict")),
        }
    }

    fn complete(
        &self,
        instruction: &str,
        user: &str,
        schema: Value,
        max_tokens: u32,
    ) -> Result<Value> {
        let format = json!({"type":"json_schema","json_schema":{"name":"result","strict":true,"schema":schema}});
        let payload = json!({"model":"memoryh-qwen3.5-2b","messages":[
            {"role":"system","content":instruction},{"role":"user","content":user}],
            "response_format":format,"temperature":0,"seed":42,"max_tokens":max_tokens,"cache_prompt":false,
            "chat_template_kwargs":{"enable_thinking":false}});
        let response: Value = self
            .agent
            .post(&self.url)
            .send_json(payload)
            .map_err(split_error)?
            .body_mut()
            .read_json()
            .map_err(split_error)?;
        let choice = &response["choices"][0];
        if choice["finish_reason"] != "stop" {
            return Err(split_error("incomplete response"));
        }
        let body = choice["message"]["content"]
            .as_str()
            .ok_or_else(|| split_error("missing content"))?;
        serde_json::from_str(body).map_err(split_error)
    }
}

fn split_error(err: impl std::fmt::Display) -> Error {
    Error::InvalidQuery(format!("Qwen3.5-2B splitter: {err}"))
}

// Bound each extraction prompt without silently dropping the rest of a long original.
fn chunks(mut body: &str) -> impl Iterator<Item = &str> {
    std::iter::from_fn(move || {
        if body.is_empty() {
            return None;
        }
        let mut end = 1200.min(body.len());
        while !body.is_char_boundary(end) {
            end -= 1;
        }
        if end < body.len()
            && let Some((i, c)) = body[..end]
                .rmatch_indices(['。', '！', '？', '.', '!', '?', '\n'])
                .next()
        {
            end = i + c.len();
        }
        let (chunk, rest) = body.split_at(end);
        body = rest;
        Some(chunk)
    })
}

// Only resolve time expressions the extractor can point to in the event's original sentence.
fn occurrence_day(span: &str, quote: &str, session: Option<Date>) -> Option<Date> {
    let span = span.trim();
    if span.is_empty() || !quote.contains(span) {
        return None;
    }
    let text = span.to_lowercase();
    if let Some(date) = session_date(&text) {
        return Some(date);
    }
    let day = session?;
    for (prefix, offset) in [
        ("今天", 0),
        ("昨天", -1),
        ("前天", -2),
        ("明天", 1),
        ("后天", 2),
        ("today", 0),
        ("yesterday", -1),
        ("tomorrow", 1),
        ("last night", -1),
    ] {
        if let Some(suffix) = text.strip_prefix(prefix) {
            // Allow a time of day, but not an interval such as “昨天到今天”.
            if [
                "",
                "早上",
                "早晨",
                "上午",
                "中午",
                "下午",
                "傍晚",
                "晚上",
                "夜里",
                " morning",
                " afternoon",
                " evening",
                " night",
            ]
            .contains(&suffix)
            {
                return day.checked_add(time::Duration::days(offset));
            }
        }
    }
    None
}

fn grounded_fact<'a>(item: &'a Value, part: &str) -> Option<(&'a str, &'a str)> {
    let quote = item["quote"].as_str()?.trim();
    let mention = item["mention"].as_str()?.trim();
    let label = item["label"].as_str()?.trim();
    // A generic generated class such as “家具” can collapse unrelated objects.
    // Use a more specific label only when the original quote supports it verbatim.
    let name = if label.contains(mention) && quote.contains(label) {
        label
    } else {
        mention
    };
    (item["status"] == "confirmed"
        && !mention.is_empty()
        && quote.contains(mention)
        && part.contains(quote)
        && name.len() <= 180)
        .then_some((name, quote))
}

fn parse(query: &str, decoded: &Value) -> Result<Plan> {
    let intent = match decoded["intent"].as_str() {
        Some("lookup") => QueryIntent::Lookup,
        Some("recommendation") => QueryIntent::Recommendation,
        Some("inventory") => QueryIntent::Inventory,
        _ => return Err(split_error("invalid query intent")),
    };
    let list = decoded["subquestions"]
        .as_array()
        .ok_or_else(|| split_error("expected subquestions list"))?;
    if list.len() > 4 {
        return Err(split_error("too many subquestions"));
    }
    let mut clauses = Vec::new();
    for item in list {
        let clause = item
            .as_str()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| split_error("empty or invalid subquestion"))?;
        if clause != query && !clauses.contains(&clause) {
            clauses.push(clause);
        }
    }
    let mut variants = vec![query.to_owned()];
    if clauses.len() > 1 {
        variants.extend(clauses.into_iter().map(str::to_owned));
    }
    Ok(Plan { intent, variants })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn extracted_fact_requires_an_observed_mention_in_quoted_evidence() {
        let memory = "上月买了一盏台灯。";
        let mut item = json!({"mention":"台灯","quote":"上月买了一盏台灯","status":"confirmed","label":"书桌台灯"});
        assert_eq!(
            grounded_fact(&item, memory),
            Some(("台灯", "上月买了一盏台灯"))
        );
        item["mention"] = json!("吊灯");
        assert!(grounded_fact(&item, memory).is_none());
        item["mention"] = json!("台灯");
        item["status"] = json!("excluded");
        assert!(grounded_fact(&item, memory).is_none());
        let long = "转了一圈，看到了日出。".repeat(170);
        assert_eq!(chunks(&long).collect::<String>(), long);
        assert!(chunks(&long).all(|chunk| chunk.len() <= 1200));
    }

    #[test]
    fn repeated_events_use_the_occurrence_day_in_the_quote() {
        let day = session_date("2025-06-02");
        assert_eq!(
            occurrence_day("昨天早上", "昨天早上去了公园", day),
            session_date("2025-06-01")
        );
        assert_eq!(occurrence_day("今天", "今天又去了公园", day), day);
        assert!(occurrence_day("昨天", "下次再去", day).is_none());
        assert!(occurrence_day("昨天到今天", "昨天到今天", day).is_none());
    }

    #[test]
    fn retains_whole_query_and_distinct_subquestions() {
        let result = parse(
            "整题",
            &json!({"intent":"lookup","subquestions":["甲","乙","甲"]}),
        )
        .unwrap();
        assert_eq!(result.variants, ["整题", "甲", "乙"]);
        assert!(parse("整题", &json!({})).is_err());
    }
}
