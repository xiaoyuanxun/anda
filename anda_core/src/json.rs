use schemars::{
    JsonSchema, Schema,
    generate::SchemaSettings,
    transform::{RestrictFormats, Transform},
};

/// Generate JSON schema for a given type T.
pub fn root_schema_for<T: JsonSchema>() -> Schema {
    let settings = SchemaSettings::draft2020_12().with(|s| {
        s.inline_subschemas = true;
        s.meta_schema = None; // Remove the $schema field

        let mut formater = RestrictFormats::default();
        formater.infer_from_meta_schema = false; // Do not infer formats from meta schema
        s.transforms.push(Box::new(formater)); // Remove the $format field
    });
    let generator = settings.into_generator();
    generator.into_root_schema_for::<T>()
}

/// Generate JSON schema for a given type T. Returns as serde_json::Value.
pub fn gen_schema_for<T: JsonSchema>() -> serde_json::Value {
    root_schema_for::<T>().to_value()
}

// Function Calling has strict requirements for JsonSchema, use fix_json_schema to fix it.
// 1. Remove $schema field;
// 2. Remove $format field;
// 3. Object type Schema must set additionalProperties: false;
// 4. required field should include all properties fields, meaning all struct fields are required (no Option).
// pub fn fix_json_schema(schema: &mut Schema) {
//     schema.meta_schema = None; // Remove the $schema field
//     fix_obj_schema(&mut schema.schema);
// }

// fn fix_obj_schema(schema: &mut SchemaObject) {
//     schema.format = None; // Remove the $format field
//     if let Some(obj) = &mut schema.object {
//         // https://platform.openai.com/docs/guides/structured-outputs#additionalproperties-false-must-always-be-set-in-objects
//         obj.additional_properties = Some(Box::new(Schema::Bool(false)));
//         if obj.required.len() != obj.properties.len() {
//             obj.required = obj.properties.keys().cloned().collect();
//         }
//         for v in obj.properties.values_mut() {
//             if let Schema::Object(o) = v {
//                 fix_obj_schema(o);
//             }
//         }
//     }
//     if let Some(arr) = &mut schema.array {
//         if let Some(v) = &mut arr.items {
//             match v {
//                 SingleOrVec::Single(v) => {
//                     if let Schema::Object(o) = v.as_mut() {
//                         fix_obj_schema(o);
//                     }
//                 }
//                 SingleOrVec::Vec(arr) => {
//                     for v in arr {
//                         if let Schema::Object(o) = v {
//                             fix_obj_schema(o);
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }
