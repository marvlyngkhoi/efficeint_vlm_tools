import os
import json

def is_correct(model_output, reference_answers):
    # Clean prediction: remove asterisks, strip whitespace, lowercase
    model_output = model_output.replace("*", "").strip().lower()
    normalized_refs = [ref.strip().lower() for ref in reference_answers]
    return model_output in normalized_refs

def evaluate_and_save_avg_flat(directory_path, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    for filename in os.listdir(directory_path):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(directory_path, filename)

        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data_list = json.load(f)
            except Exception as e:
                print(f"Skipping {filename} due to JSON load error: {e}")
                continue

        if not isinstance(data_list, list):
            print(f"Skipping {filename} — expected a list of entries.")
            continue

        # Initialize sums
        total_accuracy = 0
        total_preprocess = 0
        total_generation = 0
        total_total_time = 0
        total_memory = 0
        total_peak_memory = 0
        count = 0

        for data in data_list:
            references = data.get("reference_answer", [])
            prediction = data.get("model_output", "").strip()
            metrics = data.get("metrics", {})

            total_accuracy += 1 if is_correct(prediction, references) else 0
            total_preprocess += metrics.get("preprocess_time_sec", 0)
            total_generation += metrics.get("generation_time_sec", 0)
            total_total_time += metrics.get("total_time_sec", 0)
            total_memory += metrics.get("memory_used_gb", 0)
            total_peak_memory += metrics.get("peak_memory_gb", 0)
            count += 1

        if count == 0:
            continue

        summary = {
            "model_name": filename.split('.json')[0],
            "average_accuracy": total_accuracy / count,
            "preprocess_time_sec": total_preprocess / count,
            "generation_time_sec": total_generation / count,
            "total_time_sec": total_total_time / count,
            "memory_used_gb": total_memory / count,
            "peak_memory_gb": total_peak_memory / count
        }

        out_file_path = os.path.join(output_directory, filename)
        with open(out_file_path, 'w', encoding='utf-8') as f_out:
            json.dump(summary, f_out, indent=4)

    print(f"✅ Summary averages saved in: {output_directory}")

# Example usage
evaluate_and_save_avg_flat("eval/", "eval_reports/")
