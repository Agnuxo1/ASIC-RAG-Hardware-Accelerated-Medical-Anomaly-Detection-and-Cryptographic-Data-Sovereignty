from datasets import load_dataset

def inspect():
    try:
        print("Loading dataset...")
        dataset = load_dataset("mmenendezg/pneumonia_x_ray", name="default")
        print("\nDataset Structure:")
        print(dataset)
        
        print("\nFirst Item Keys:")
        print(dataset['train'][0].keys())
        
        print("\nFeatures:")
        print(dataset['train'].features)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect()
