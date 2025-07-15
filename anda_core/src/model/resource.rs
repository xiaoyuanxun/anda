pub use anda_db_schema::Resource;

/// Extracts resources with the given tags from the list of resources.
pub fn select_resources(resources: &mut Vec<Resource>, tags: &[&str]) -> Option<Vec<Resource>> {
    if tags.is_empty() {
        return None;
    }

    if tags.first() == Some(&"*") {
        return Some(std::mem::take(resources));
    }

    // nightly feature:
    // {
    //     let res: Vec<Resource> = resources
    //         .extract_if(.., |r| tags.contains(&r.tag.as_str()))
    //         .collect();
    //     if res.is_empty() { None } else { Some(res) }
    // }

    {
        let mut res = Vec::new();
        let mut i = 0;

        while i < resources.len() {
            if tags.contains(&resources[i].tag.as_str()) {
                res.push(resources.remove(i));
            } else {
                i += 1;
            }
        }
        if res.is_empty() { None } else { Some(res) }
    }
}
