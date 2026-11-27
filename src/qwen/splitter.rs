use crate::{Error, Result};
use serde_json::{Value, json};
use std::{net::SocketAddr, time::Duration};

const INSTRUCTION: &str = "你是记忆检索系统的问题分解器。判断用户是否同时询问多个可以独立检索的信息。单一意图返回空数组。多个独立意图拆成二至四个自足的中文检索问题，保留原有对象、时间、地点和限制；只拆问题，不补充事实，不作答。输出对象的 subquestions 字段。";

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
    pub(super) fn variants(&self, query: &str) -> Result<Vec<String>> {
        let format = json!({"type":"json_schema","json_schema":{"name":"decomposition","strict":true,"schema":{
            "type":"object","properties":{"subquestions":{"type":"array","items":{"type":"string"},"maxItems":4}},
            "required":["subquestions"],"additionalProperties":false}}});
        let payload = json!({"model":"memoryh-qwen3.5-2b","messages":[
            {"role":"system","content":INSTRUCTION},{"role":"user","content":query}],
            "response_format":format,"temperature":0,"seed":42,"max_tokens":256,"cache_prompt":false,
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
        parse(query, body)
    }
}

fn split_error(err: impl std::fmt::Display) -> Error {
    Error::InvalidQuery(format!("Qwen3.5-2B splitter: {err}"))
}

fn parse(query: &str, body: &str) -> Result<Vec<String>> {
    let decoded: Value = serde_json::from_str(body).map_err(split_error)?;
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
    Ok(variants)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retains_whole_query_and_distinct_subquestions() {
        assert_eq!(parse("整题", r#"{"subquestions":[]}"#).unwrap(), ["整题"]);
        assert_eq!(
            parse("整题", r#"{"subquestions":["甲","乙","甲"]}"#).unwrap(),
            ["整题", "甲", "乙"]
        );
        assert!(parse("整题", "{}").is_err());
    }
}
