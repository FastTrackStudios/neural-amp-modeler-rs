//! Activation functions, numerically identical to NeuralAmpModelerCore's
//! `NAM/activations.h`.

use serde_json::Value;

#[derive(Clone, Debug)]
pub(crate) enum Activation {
    Tanh,
    Hardtanh,
    Fasttanh,
    ReLU,
    LeakyReLU(f32),
    /// PReLU with one or more negative slopes (applied `pos % len`, matching
    /// the C++ flat-array `apply`).
    PReLU(Vec<f32>),
    Sigmoid,
    SiLU,
    Hardswish,
    LeakyHardtanh {
        min_val: f32,
        max_val: f32,
        min_slope: f32,
        max_slope: f32,
    },
    Softsign,
}

#[inline]
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn fast_tanh(x: f32) -> f32 {
    let ax = x.abs();
    let x2 = x * x;
    (x * (2.455_507_5 + 2.455_507_5 * ax + (0.893_229_9 + 0.821_226_7 * ax) * x2))
        / (2.445_066_3 + (2.445_066_3 + x2) * (x + 0.814_642_7 * x * ax).abs())
}

impl Activation {
    #[inline]
    fn scalar(&self, x: f32, pos: usize) -> f32 {
        match self {
            Activation::Tanh => x.tanh(),
            Activation::Hardtanh => x.clamp(-1.0, 1.0),
            Activation::Fasttanh => fast_tanh(x),
            Activation::ReLU => {
                if x > 0.0 {
                    x
                } else {
                    0.0
                }
            }
            Activation::LeakyReLU(slope) => {
                if x > 0.0 {
                    x
                } else {
                    slope * x
                }
            }
            Activation::PReLU(slopes) => {
                if x > 0.0 {
                    x
                } else {
                    slopes[pos % slopes.len()] * x
                }
            }
            Activation::Sigmoid => sigmoid(x),
            Activation::SiLU => x * sigmoid(x),
            Activation::Hardswish => {
                let t = (x + 3.0).clamp(0.0, 6.0);
                x * t * (1.0 / 6.0)
            }
            Activation::LeakyHardtanh {
                min_val,
                max_val,
                min_slope,
                max_slope,
            } => {
                if x < *min_val {
                    (x - min_val) * min_slope + min_val
                } else if x > *max_val {
                    (x - max_val) * max_slope + max_val
                } else {
                    x
                }
            }
            Activation::Softsign => x / (1.0 + x.abs()),
        }
    }

    /// In-place apply over a flat (column-major-contiguous) buffer.
    pub fn apply(&self, data: &mut [f32]) {
        for (pos, v) in data.iter_mut().enumerate() {
            *v = self.scalar(*v, pos);
        }
    }

    /// Parse from a .nam activation config: either a string ("Tanh") or an
    /// object ({"type": "LeakyReLU", "negative_slope": 0.01}).
    pub fn from_json(j: &Value) -> Result<Self, String> {
        let type_str = if let Some(s) = j.as_str() {
            s.to_string()
        } else if let Some(obj) = j.as_object() {
            obj.get("type")
                .and_then(|t| t.as_str())
                .ok_or_else(|| "activation object missing 'type'".to_string())?
                .to_string()
        } else {
            return Err(format!("invalid activation config: {j}"));
        };

        let f32_field = |key: &str| -> Option<f32> {
            j.as_object()
                .and_then(|o| o.get(key))
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
        };

        Ok(match type_str.as_str() {
            "Tanh" => Activation::Tanh,
            "Hardtanh" => Activation::Hardtanh,
            "Fasttanh" => Activation::Fasttanh,
            "ReLU" => Activation::ReLU,
            "LeakyReLU" => Activation::LeakyReLU(f32_field("negative_slope").unwrap_or(0.01)),
            "PReLU" => {
                let slopes = if let Some(arr) = j
                    .as_object()
                    .and_then(|o| o.get("negative_slopes"))
                    .and_then(|v| v.as_array())
                {
                    arr.iter()
                        .map(|v| v.as_f64().unwrap_or(0.01) as f32)
                        .collect()
                } else {
                    vec![f32_field("negative_slope").unwrap_or(0.01)]
                };
                Activation::PReLU(slopes)
            }
            "Sigmoid" => Activation::Sigmoid,
            "SiLU" => Activation::SiLU,
            "Hardswish" => Activation::Hardswish,
            "LeakyHardtanh" | "LeakyHardTanh" => Activation::LeakyHardtanh {
                min_val: f32_field("min_val").unwrap_or(-1.0),
                max_val: f32_field("max_val").unwrap_or(1.0),
                min_slope: f32_field("min_slope").unwrap_or(0.01),
                max_slope: f32_field("max_slope").unwrap_or(0.01),
            },
            "Softsign" => Activation::Softsign,
            other => return Err(format!("unknown activation type: {other}")),
        })
    }
}
