use std::collections::{BTreeMap, HashMap, HashSet};

use unicode_normalization::UnicodeNormalization;

use crate::{Error, Result};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum TreeLayout {
    #[default]
    Semantic,
    /// Two levels below the root. Each leaf matches the first component of a logical key.
    Prefixes(Vec<Vec<PrefixLeaf>>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixLeaf {
    pub label: String,
    pub key_prefix: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchMethod {
    Keywords,
    Flat,
    Tree,
}

#[derive(Clone, Debug)]
pub struct SearchOptions {
    pub top_k: usize,
    pub beam_width: usize,
    pub tree: TreeLayout,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            top_k: 3,
            beam_width: 2,
            tree: TreeLayout::Semantic,
        }
    }
}

pub struct Query<'a> {
    pub text: &'a str,
    pub embedding: Option<&'a [f32]>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Ranking {
    pub lexical: Vec<String>,
    pub flat: Vec<String>,
    pub tree: Vec<String>,
    pub flat_vector: Vec<String>,
    pub tree_vector: Vec<String>,
    pub routes: Vec<String>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct SearchResult {
    pub whole: Ranking,
    pub parts: Vec<Ranking>,
    pub flat_split: Vec<String>,
    pub tree_split: Vec<String>,
    pub flat_vector_first: Vec<String>,
    pub tree_vector_first: Vec<String>,
}

#[derive(Default)]
pub(crate) struct RetrievalIndex {
    documents: BTreeMap<String, Document>,
    postings: HashMap<String, BTreeMap<String, usize>>,
    total_terms: usize,
    pub(crate) dimension: Option<usize>,
    tree: Option<(TreeLayout, Tree)>,
}

struct Document {
    key: String,
    terms: HashMap<String, usize>,
    len: usize,
    embedding: Option<Vec<f32>>,
}

impl RetrievalIndex {
    pub(crate) fn insert(&mut self, id: &str, key: &str, body: &str, embedding: Option<Vec<f32>>) {
        let mut terms = HashMap::new();
        let mut len = 0;
        for term in tokenize(body) {
            *terms.entry(term.to_ascii_lowercase()).or_insert(0) += 1;
            len += 1;
        }
        for (term, count) in &terms {
            self.postings
                .entry(term.clone())
                .or_default()
                .insert(id.to_owned(), *count);
        }
        self.total_terms += len;
        if let Some(vector) = &embedding {
            self.dimension = Some(vector.len());
        }
        self.documents.insert(
            id.to_owned(),
            Document {
                key: key.into(),
                terms,
                len,
                embedding,
            },
        );
        self.tree = None;
    }

    pub(crate) fn remove(&mut self, id: &str) {
        if let Some(document) = self.documents.remove(id) {
            self.total_terms -= document.len;
            for term in document.terms.keys() {
                let postings = self.postings.get_mut(term).expect("indexed document term");
                postings.remove(id);
                if postings.is_empty() {
                    self.postings.remove(term);
                }
            }
            self.tree = None;
        }
    }

    pub(crate) fn lexical(&self, text: &str, limit: usize) -> Vec<String> {
        if self.total_terms == 0 || limit == 0 {
            return Vec::new();
        }
        let count = self.documents.len() as f64;
        let average_len = self.total_terms as f64 / count;
        let mut scores = HashMap::new();
        let mut terms = HashSet::new();
        for term in tokenize(text) {
            if !terms.insert(term.clone()) {
                continue;
            }
            let Some(postings) = self.postings.get(&term.to_ascii_lowercase()) else {
                continue;
            };
            let matched = postings.len() as f64;
            let idf = ((count - matched + 0.5) / (matched + 0.5)).ln();
            let idf = if idf > 0.0 { idf } else { 1e-6 };
            for (id, &frequency) in postings {
                let frequency = frequency as f64;
                let len = self.documents[id].len as f64;
                let score =
                    idf * (frequency * 2.2) / (frequency + 1.2 * (0.25 + 0.75 * len / average_len));
                *scores.entry(id).or_insert(0.0) += score;
            }
        }
        ranked(scores, limit)
    }

    pub(crate) fn search(
        &mut self,
        queries: &[Query<'_>],
        options: &SearchOptions,
    ) -> Result<SearchResult> {
        let Some(question) = queries.first() else {
            return Err(Error::InvalidQuery("missing question".into()));
        };
        if options.beam_width == 0 {
            return Err(Error::InvalidQuery("beam width must be positive".into()));
        }
        let embeddings: Vec<_> = queries
            .iter()
            .map(|query| {
                query
                    .embedding
                    .map(|vector| {
                        if self.dimension.is_some_and(|size| size != vector.len()) {
                            return Err(Error::InvalidQuery(
                                "embedding dimension does not match the store".into(),
                            ));
                        }
                        normalize(vector.iter().copied()).ok_or_else(|| {
                            Error::InvalidQuery("embedding must be finite and nonzero".into())
                        })
                    })
                    .transpose()
            })
            .collect::<Result<_>>()?;
        let vectors: Vec<_> = embeddings.iter().flatten().collect();
        if !vectors.is_empty()
            && self
                .tree
                .as_ref()
                .is_none_or(|(layout, _)| *layout != options.tree)
        {
            self.tree = Some((
                options.tree.clone(),
                Tree::build(&self.documents, &options.tree),
            ));
        }
        let whole = self.rank(question.text, &vectors, options);
        let parts: Vec<_> = queries[1..]
            .iter()
            .zip(&embeddings[1..])
            .map(|(query, embedding)| {
                self.rank(query.text, &embedding.iter().collect::<Vec<_>>(), options)
            })
            .collect();
        let (flat_split, flat_vector_first) = merge_variants(
            &whole.flat,
            &parts,
            |part| (&part.flat, &part.flat_vector),
            options.top_k,
        );
        let (tree_split, tree_vector_first) = merge_variants(
            &whole.tree,
            &parts,
            |part| (&part.tree, &part.tree_vector),
            options.top_k,
        );
        Ok(SearchResult {
            whole,
            parts,
            flat_split,
            tree_split,
            flat_vector_first,
            tree_vector_first,
        })
    }

    fn rank(&self, text: &str, vectors: &[&Vec<f32>], options: &SearchOptions) -> Ranking {
        let limit = options.top_k.saturating_mul(3);
        let lexical = self.lexical(text, limit);
        let mut routes = Vec::new();
        let mut tree_ids = Vec::new();
        if let Some((_, tree)) = &self.tree {
            let mut seen = HashSet::new();
            for vector in vectors {
                for leaf in tree.route(vector, options.beam_width) {
                    if seen.insert(&leaf.label) {
                        routes.push(leaf.label.clone());
                        tree_ids.extend(leaf.ids.iter());
                    }
                }
            }
        }
        let flat_vector = self.rank_vectors(self.documents.keys(), vectors);
        let tree_vector = self.rank_vectors(tree_ids.into_iter(), vectors);
        let flat = fuse(&lexical, &flat_vector, limit);
        let tree = fuse(&lexical, &tree_vector, limit);
        Ranking {
            lexical,
            flat,
            tree,
            flat_vector,
            tree_vector,
            routes,
        }
    }

    fn rank_vectors<'a>(
        &self,
        ids: impl Iterator<Item = &'a String>,
        vectors: &[&Vec<f32>],
    ) -> Vec<String> {
        if vectors.is_empty() {
            return Vec::new();
        }
        let scores = ids.filter_map(|id| {
            let candidate = self.documents[id].embedding.as_ref()?;
            let score = vectors
                .iter()
                .map(|query| dot(query, candidate))
                .max_by(f32::total_cmp)?;
            Some((id, f64::from(score)))
        });
        ranked(scores, usize::MAX)
    }
}

fn tokenize(text: &str) -> Vec<String> {
    fn chinese(c: char) -> bool {
        ('\u{3400}'..='\u{9fff}').contains(&c)
    }

    let mut terms = Vec::new();
    let mut chars = text.nfkc().peekable();
    while let Some(c) = chars.next() {
        if c.is_ascii_alphanumeric() {
            let mut term = String::from(c);
            while chars.peek().is_some_and(char::is_ascii_alphanumeric) {
                term.push(chars.next().expect("peeked character"));
            }
            terms.push(term);
        } else if chinese(c) {
            let mut previous = c;
            let mut has_pair = false;
            while chars.peek().is_some_and(|c| chinese(*c)) {
                let next = chars.next().expect("peeked character");
                terms.push([previous, next].into_iter().collect());
                previous = next;
                has_pair = true;
            }
            if !has_pair {
                terms.push(c.to_string());
            }
        }
    }
    terms
}

/// Produce the whole question and, when present, distinct substantive clauses.
pub fn query_variants(question: &str) -> Vec<&str> {
    let mut clauses = Vec::new();
    let mut start = 0;
    for (index, c) in question.char_indices() {
        if index < start {
            continue;
        }
        let delimiter = if "？?；;。！!".contains(c) {
            c.len_utf8()
        } else {
            ["另外", "以及", "并且", "还有"]
                .into_iter()
                .find(|word| question[index..].starts_with(word))
                .map_or(0, str::len)
        };
        if delimiter > 0 {
            clauses.push(&question[start..index]);
            start = index + delimiter;
        }
    }
    clauses.push(&question[start..]);
    clauses = clauses
        .into_iter()
        .map(|text| text.trim_matches([' ', '，', ',', '、']))
        .filter(|text| text.chars().take(4).count() == 4)
        .collect();
    let mut variants = vec![question];
    if clauses.len() > 1 {
        let mut seen = HashSet::new();
        variants.extend(clauses.into_iter().filter(|clause| seen.insert(*clause)));
    }
    variants
}

pub(crate) fn normalize(values: impl Iterator<Item = f32>) -> Option<Vec<f32>> {
    let vector: Vec<_> = values.collect();
    let norm = vector
        .iter()
        .map(|&v| f64::from(v).powi(2))
        .sum::<f64>()
        .sqrt();
    (norm > 0.0 && norm.is_finite()).then(|| {
        vector
            .into_iter()
            .map(|v| (f64::from(v) / norm) as f32)
            .collect()
    })
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn ranked<'a>(scores: impl IntoIterator<Item = (&'a String, f64)>, limit: usize) -> Vec<String> {
    let mut scores: Vec<_> = scores.into_iter().collect();
    scores.sort_unstable_by(|(left, a), (right, b)| b.total_cmp(a).then_with(|| left.cmp(right)));
    scores
        .into_iter()
        .take(limit)
        .map(|(id, _)| id.clone())
        .collect()
}

fn fuse(lexical: &[String], vector: &[String], limit: usize) -> Vec<String> {
    let mut scores = HashMap::new();
    for ranking in [lexical, vector] {
        for (index, id) in ranking.iter().take(limit).enumerate() {
            *scores.entry(id).or_insert(0.0) += 1.0 / (61 + index) as f64;
        }
    }
    ranked(scores, usize::MAX)
}

fn merge_variants(
    whole: &[String],
    parts: &[Ranking],
    select: impl Fn(&Ranking) -> (&[String], &[String]),
    limit: usize,
) -> (Vec<String>, Vec<String>) {
    if parts.is_empty() {
        return (whole.to_vec(), whole.to_vec());
    }
    let parts: Vec<_> = parts.iter().map(select).collect();
    (merge(&parts, false, limit), merge(&parts, true, limit))
}

fn merge(parts: &[(&[String], &[String])], reserve: bool, limit: usize) -> Vec<String> {
    let mut seen = HashSet::new();
    let first = parts.iter().filter_map(|(_, vectors)| vectors.first());
    let mut result: Vec<_> = if reserve {
        first
            .filter(|id| seen.insert(*id))
            .take(limit)
            .cloned()
            .collect()
    } else {
        Vec::new()
    };
    let mut queues: Vec<_> = parts.iter().map(|(ranking, _)| ranking.iter()).collect();
    while result.len() < limit {
        let before = result.len();
        for queue in &mut queues {
            if let Some(id) = queue.find(|id| seen.insert(*id)) {
                result.push(id.clone());
            }
            if result.len() == limit {
                break;
            }
        }
        if before == result.len() {
            break;
        }
    }
    result
}

struct Tree {
    branches: Vec<Branch>,
}

struct Branch {
    centroid: Vec<f32>,
    leaves: Vec<Leaf>,
}

struct Leaf {
    label: String,
    centroid: Vec<f32>,
    ids: Vec<String>,
}

impl Tree {
    fn build(documents: &BTreeMap<String, Document>, layout: &TreeLayout) -> Self {
        let vectors: BTreeMap<_, _> = documents
            .iter()
            .filter_map(|(id, document)| {
                document
                    .embedding
                    .as_deref()
                    .map(|vector| (id.as_str(), vector))
            })
            .collect();
        let groups: Vec<_> = match layout {
            TreeLayout::Semantic => {
                let ids = vectors.keys().copied().collect();
                partition(ids, &vectors)
                    .into_iter()
                    .enumerate()
                    .map(|(branch, ids)| {
                        partition(ids, &vectors)
                            .into_iter()
                            .enumerate()
                            .map(|(leaf, ids)| (format!("{branch}{leaf}"), ids))
                            .collect()
                    })
                    .collect()
            }
            TreeLayout::Prefixes(branches) => branches
                .iter()
                .map(|branch| {
                    branch
                        .iter()
                        .map(|leaf| {
                            let ids = vectors
                                .keys()
                                .copied()
                                .filter(|id| {
                                    documents[*id].key.split('.').next()
                                        == Some(leaf.key_prefix.as_str())
                                })
                                .collect();
                            (leaf.label.clone(), ids)
                        })
                        .collect()
                })
                .collect(),
        };
        let branches = groups
            .into_iter()
            .filter_map(|leaves: Vec<(_, Vec<_>)>| {
                let ids: Vec<_> = leaves
                    .iter()
                    .flat_map(|(_, ids)| ids.iter().copied())
                    .collect();
                let branch_centroid = centroid(&ids, &vectors)?;
                let leaves = leaves
                    .into_iter()
                    .filter_map(|(label, ids)| {
                        let centroid = centroid(&ids, &vectors)?;
                        Some(Leaf {
                            label,
                            centroid,
                            ids: ids.into_iter().map(str::to_owned).collect(),
                        })
                    })
                    .collect();
                Some(Branch {
                    centroid: branch_centroid,
                    leaves,
                })
            })
            .collect();
        Self { branches }
    }

    fn route<'a>(&'a self, query: &[f32], beam_width: usize) -> Vec<&'a Leaf> {
        let mut frontier: Vec<_> = self
            .branches
            .iter()
            .map(|node| (node, f64::from(dot(query, &node.centroid))))
            .collect();
        frontier.sort_by(|(_, a), (_, b)| b.total_cmp(a));
        frontier.truncate(beam_width);
        let mut candidates: Vec<_> = frontier
            .into_iter()
            .flat_map(|(branch, score)| {
                branch
                    .leaves
                    .iter()
                    .map(move |leaf| (leaf, score + f64::from(dot(query, &leaf.centroid))))
            })
            .collect();
        candidates.sort_by(|(_, a), (_, b)| b.total_cmp(a));
        candidates
            .into_iter()
            .take(beam_width)
            .map(|(leaf, _)| leaf)
            .collect()
    }
}

fn summed(ids: &[&str], vectors: &BTreeMap<&str, &[f32]>) -> Option<Vec<f32>> {
    let mut sum = vectors[*ids.first()?].to_vec();
    for id in &ids[1..] {
        for (total, value) in sum.iter_mut().zip(vectors[id]) {
            *total += value;
        }
    }
    Some(sum)
}

fn centroid(ids: &[&str], vectors: &BTreeMap<&str, &[f32]>) -> Option<Vec<f32>> {
    let sum = summed(ids, vectors)?;
    // Opposing vectors can cancel. A zero centroid keeps the branch routable.
    Some(normalize(sum.iter().copied()).unwrap_or(sum))
}

fn partition<'a>(mut ids: Vec<&'a str>, vectors: &BTreeMap<&str, &[f32]>) -> [Vec<&'a str>; 2] {
    if ids.len() < 2 {
        return [ids, Vec::new()];
    }
    let mut center = summed(&ids, vectors).expect("nonempty partition");
    for value in &mut center {
        *value /= ids.len() as f32;
    }
    let farthest = |axis: &[f32]| {
        ids.iter()
            .copied()
            .min_by(|left, right| {
                dot(vectors[left], axis)
                    .total_cmp(&dot(vectors[right], axis))
                    .then_with(|| left.cmp(right))
            })
            .expect("nonempty partition")
    };
    let first = vectors[farthest(&center)];
    let second = vectors[farthest(first)];
    let axis: Vec<_> = second.iter().zip(first).map(|(b, a)| b - a).collect();
    ids.sort_unstable_by(|left, right| {
        dot(vectors[left], &axis)
            .total_cmp(&dot(vectors[right], &axis))
            .then_with(|| left.cmp(right))
    });
    let right = ids.split_off(ids.len() / 2);
    [ids, right]
}
