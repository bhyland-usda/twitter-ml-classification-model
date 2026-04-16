# Twitter ML Classification Model

This project is a classical ML model trainer written in Rust to be trained
on raw Twitter post comments exported to a CSV file. The output is a model
that can classify each comment to be actionble or non-actionable.

Once the model is created the Python scripts use the Typescript guardrails file
to do more precise finetuning so that the model can be as accurate as possible.

## Tech Stack
- Languages: Rust, Python, and Typescript
- Framework: Candle for inference
