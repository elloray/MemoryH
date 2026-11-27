use crate::Record;
use std::collections::{BTreeSet, HashMap, HashSet};
pub use time::Date;

/// Dates belong to the question and the original session, independently of storage revisions.
#[derive(Default)]
pub struct QueryContext {
    pub question_date: Option<Date>,
    /// Explicit session dates keyed by immutable record ID take precedence over a dated source.
    pub session_dates: HashMap<String, Date>,
}

/// Accept ISO and LongMemEval date prefixes, including a following time or weekday.
pub fn session_date(source: &str) -> Option<Date> {
    let source = source.trim_start();
    let prefix = source.get(..10)?;
    let separator = prefix.as_bytes()[4];
    if !matches!(separator, b'/' | b'-')
        || prefix.as_bytes()[7] != separator
        || source
            .as_bytes()
            .get(10)
            .is_some_and(|b| !matches!(b, b' ' | b'T' | b'(' | b'\t'))
    {
        return None;
    }
    let year = prefix.get(..4)?.parse().ok()?;
    let month: u8 = prefix.get(5..7)?.parse().ok()?;
    let day = prefix.get(8..)?.parse().ok()?;
    Date::from_calendar_date(year, month.try_into().ok()?, day).ok()
}

impl QueryContext {
    pub(super) fn query(&self, text: &str) -> String {
        match self.question_date {
            Some(date) => format!("Question date / 提问日期: {date}\n{text}"),
            None => text.to_owned(),
        }
    }

    pub(super) fn date_of(&self, record: &Record) -> Option<Date> {
        self.session_dates
            .get(&record.memory.id)
            .copied()
            .or_else(|| session_date(&record.memory.source))
    }

    pub(super) fn document(&self, record: &Record) -> String {
        let Some(date) = self.date_of(record) else {
            return record.memory.body.clone();
        };
        let offset = self
            .question_date
            .map(|asked| {
                let days = (asked - date).whole_days();
                match days.cmp(&0) {
                    std::cmp::Ordering::Greater => {
                        format!(" ({days} days before question / 提问前 {days} 天)")
                    }
                    std::cmp::Ordering::Equal => " (question day / 提问当天)".into(),
                    std::cmp::Ordering::Less => {
                        format!(" ({} days after question / 提问后 {} 天)", -days, -days)
                    }
                }
            })
            .unwrap_or_default();
        format!(
            "Session date / 会话日期: {date}{offset}\n{}",
            record.memory.body
        )
    }
}

/// Reserve the strongest evidence for each independently scored aspect. The whole-query
/// winner leads, then remaining slots follow the direct relevance ranking.
pub(super) fn evidence_order(whole: &[f32], aspects: &[Vec<f32>]) -> Vec<usize> {
    let mut direct: Vec<_> = (0..whole.len()).collect();
    direct.sort_by(|&a, &b| whole[b].total_cmp(&whole[a]));
    let Some(&first) = direct.first() else {
        return direct;
    };
    let mut seen = HashSet::new();
    let mut order = vec![first];
    seen.insert(first);
    for scores in aspects {
        // In a tie prefer the document ranked higher against the whole question.
        let mut winner = first;
        for &i in &direct {
            if scores[i] > scores[winner] {
                winner = i;
            }
        }
        if seen.insert(winner) {
            order.push(winner);
        }
    }
    order.extend(direct.into_iter().filter(|i| !seen.contains(i)));
    order
}

/// Greedy set coverage. The input order breaks ties and fills any slots left after coverage.
pub(super) fn cover_facts(
    mut order: Vec<usize>,
    facts: &[BTreeSet<String>],
    slots: usize,
) -> Vec<usize> {
    let mut covered = BTreeSet::new();
    let mut prefix = Vec::new();
    while prefix.len() < slots {
        let mut best = None;
        let mut best_gain = 0;
        for (position, &i) in order.iter().enumerate() {
            let gain = facts[i].difference(&covered).count();
            if gain > best_gain {
                best = Some(position);
                best_gain = gain;
            }
        }
        let Some(position) = best else {
            break;
        };
        let i = order.remove(position);
        covered.extend(facts[i].iter().cloned());
        prefix.push(i);
    }
    prefix.extend(order);
    prefix
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Memory;

    #[test]
    fn dates_anchor_each_text_and_handle_missing_dates_and_leap_days() {
        let record = Record {
            memory: Memory {
                id: "s".into(),
                key: "s".into(),
                source: "2024/02/28 (Wed) 09:10".into(),
                body: "今天买了书".into(),
            },
            revision: 9,
            supersedes: None,
        };
        let mut context = QueryContext {
            question_date: session_date("2024-03-10"),
            ..Default::default()
        };
        assert!(context.document(&record).contains("11 days before"));
        context
            .session_dates
            .insert("s".into(), session_date("2024-02-29").unwrap());
        assert!(context.document(&record).contains("10 days before"));
        assert!(context.query("十天前买了什么？").contains("2024-03-10"));
        assert!(session_date("2023-02-29").is_none());
        assert!(session_date("undated session").is_none());
    }

    #[test]
    fn separate_evidence_and_novel_facts_survive_top_five() {
        let direct = [9., 8., 7., 6., 5., 4., 3.];
        let order = evidence_order(
            &direct,
            &[
                vec![1., 1., 1., 1., 1., 1., 10.],
                vec![0., 1., 1., 1., 1., 9., 1.],
            ],
        );
        assert_eq!(&order[..3], &[0, 6, 5]);
        let facts =
            ["桌", "桌", "桌", "椅", "柜", "床垫", "架"].map(|f| BTreeSet::from([f.to_owned()]));
        assert_eq!(
            &cover_facts((0..7).collect(), &facts, 5)[..5],
            &[0, 3, 4, 5, 6]
        );
        let combined = [
            BTreeSet::from(["桌".into()]),
            BTreeSet::from(["床垫".into(), "桌".into()]),
            BTreeSet::from(["椅".into()]),
        ];
        assert_eq!(cover_facts(vec![0, 1, 2], &combined, 2), [1, 2, 0]);
    }
}
