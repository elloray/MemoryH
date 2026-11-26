use memoryh::{Memory, Mutation};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct Fixture<Q> {
    pub memories: Vec<InputRecord>,
    pub corrections: Vec<InputRecord>,
    pub questions: Vec<Q>,
}

#[derive(Clone, Deserialize)]
pub struct InputRecord {
    pub id: String,
    pub key: String,
    pub source: String,
    pub text: String,
    pub supersedes: Option<String>,
}

impl From<InputRecord> for Mutation {
    fn from(record: InputRecord) -> Self {
        let memory = Memory {
            id: record.id,
            key: record.key,
            source: record.source,
            body: record.text,
        };
        match record.supersedes {
            Some(expected_id) => Self::Replace {
                expected_id,
                memory,
            },
            None => Self::Put(memory),
        }
    }
}
