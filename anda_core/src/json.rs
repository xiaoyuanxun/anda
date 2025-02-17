use schemars::schema::{RootSchema, Schema, SchemaObject, SingleOrVec};

pub use schemars::{schema_for, JsonSchema};

/// Function Calling has strict requirements for JsonSchema, use fix_json_schema to fix it
/// 1. Remove $schema field
/// 2. Remove $format field
/// 3. Object type Schema must set additionalProperties: false
/// 4. required field should include all properties fields, meaning all struct fields are required (no Option)
pub fn fix_json_schema(schema: &mut RootSchema) {
    schema.meta_schema = None; // Remove the $schema field
    fix_obj_schema(&mut schema.schema);
}

fn fix_obj_schema(schema: &mut SchemaObject) {
    schema.format = None; // Remove the $format field
    if let Some(obj) = &mut schema.object {
        // https://platform.openai.com/docs/guides/structured-outputs#additionalproperties-false-must-always-be-set-in-objects
        obj.additional_properties = Some(Box::new(Schema::Bool(false)));
        // if obj.required.len() != obj.properties.len() {
        //     obj.required = obj.properties.keys().cloned().collect();
        // }
        for v in obj.properties.values_mut() {
            if let Schema::Object(o) = v {
                fix_obj_schema(o);
            }
        }
    }
    if let Some(arr) = &mut schema.array {
        if let Some(v) = &mut arr.items {
            match v {
                SingleOrVec::Single(v) => {
                    if let Schema::Object(o) = v.as_mut() {
                        fix_obj_schema(o);
                    }
                }
                SingleOrVec::Vec(arr) => {
                    for v in arr {
                        if let Schema::Object(o) = v {
                            fix_obj_schema(o);
                        }
                    }
                }
            }
        }
    }
}
