
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Configuration
class Config:
    """Centralized configuration for the model and training"""
    MAX_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    MODEL_NAME = 'distilbert-base-uncased'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Intent categories for medical questions
    INTENT_LABELS = {
        'diagnosis': 0,
        'treatment': 1,
        'prevention': 2,
        'symptoms': 3,
        'medication': 4
    }


def create_synthetic_dataset(n_samples=1000):
    """
    Creates a synthetic medical Q&A dataset for demonstration.
    In a real project, you'd load from CSV or API.
    
    Args:
        n_samples: Number of samples to generate
    
    Returns:
        DataFrame with questions and intent labels
    """
    # Sample questions for each intent category
    templates = {
        'diagnosis': [
            "What could be causing my {}?",
            "How do doctors diagnose {}?",
            "Is {} a symptom of a serious condition?",
            "What tests are needed to confirm {}?",
            "Could my {} indicate a problem with my {}?"
        ],
        'treatment': [
            "What are the treatment options for {}?",
            "How can I treat {} at home?",
            "What's the best way to manage {}?",
            "Are there natural remedies for {}?",
            "What does a doctor prescribe for {}?"
        ],
        'prevention': [
            "How can I prevent {}?",
            "What steps reduce the risk of {}?",
            "Can {} be prevented with lifestyle changes?",
            "What should I avoid to prevent {}?",
            "Is there a vaccine for {}?"
        ],
        'symptoms': [
            "What are the symptoms of {}?",
            "How do I know if I have {}?",
            "What does {} feel like?",
            "Are {} and {} related symptoms?",
            "When should I worry about {}?"
        ],
        'medication': [
            "What medication is used for {}?",
            "What are the side effects of {}?",
            "Can I take {} with {}?",
            "How long should I take {} for?",
            "Is {} safe during pregnancy?"
        ]
    }
    
    conditions = ['headache', 'fever', 'diabetes', 'asthma', 'arthritis', 
                  'hypertension', 'back pain', 'anxiety', 'insomnia', 'allergies']
    body_parts = ['heart', 'lungs', 'stomach', 'kidneys', 'liver', 'brain']
    
    data = []
    samples_per_intent = n_samples // len(templates)
    
    for intent, question_templates in templates.items():
        for _ in range(samples_per_intent):
            template = np.random.choice(question_templates)
            condition = np.random.choice(conditions)
            
            # Handle templates with multiple placeholders
            if template.count('{}') == 2:
                body_part = np.random.choice(body_parts)
                question = template.format(condition, body_part)
            else:
                question = template.format(condition)
            
            data.append({'question': question, 'intent': intent})
    
    return pd.DataFrame(data)


class MedicalQADataset(Dataset):
    """PyTorch Dataset for medical questions"""
    
    def __init__(self, questions, labels, tokenizer, max_len):
        self.questions = questions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = str(self.questions[idx])
        label = self.labels[idx]
        
        # Tokenize the question
        encoding = self.tokenizer.encode_plus(
            question,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """
    Train the model for one epoch
    
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, device):
    """
    Evaluate the model on validation/test data
    
    Returns:
        predictions and true labels
    """
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return np.array(predictions), np.array(true_labels)


def plot_confusion_matrix(y_true, y_pred, labels):
    """Visualize the confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - Medical Intent Classification')
    plt.ylabel('True Intent')
    plt.xlabel('Predicted Intent')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix saved as 'confusion_matrix.png'")


def plot_training_history(train_losses):
    """Plot training loss over epochs"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("Training history saved as 'training_history.png'")


def main():
    """Main training and evaluation pipeline"""
    print(f"Using device: {Config.DEVICE}")
    
    # Step 1: Load and prepare data
    print("\n1. Loading dataset...")
    df = create_synthetic_dataset(n_samples=1000)
    print(f"Dataset size: {len(df)} samples")
    print(f"Intent distribution:\n{df['intent'].value_counts()}\n")
    
    # Convert intent labels to numeric
    df['label'] = df['intent'].map(Config.INTENT_LABELS)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['question'].values,
        df['label'].values,
        test_size=0.2,
        random_state=SEED,
        stratify=df['label']
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}\n")
    
    # Step 2: Initialize tokenizer and create datasets
    print("2. Preparing tokenizer and datasets...")
    tokenizer = DistilBertTokenizer.from_pretrained(Config.MODEL_NAME)
    
    train_dataset = MedicalQADataset(X_train, y_train, tokenizer, Config.MAX_LEN)
    test_dataset = MedicalQADataset(X_test, y_test, tokenizer, Config.MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)
    
    # Step 3: Initialize model
    print("3. Loading pre-trained model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=len(Config.INTENT_LABELS)
    )
    model.to(Config.DEVICE)
    
    # Step 4: Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Step 5: Training loop
    print(f"\n4. Training for {Config.EPOCHS} epochs...")
    train_losses = []
    
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
        avg_loss = train_epoch(model, train_loader, optimizer, scheduler, Config.DEVICE)
        train_losses.append(avg_loss)
        print(f"Average training loss: {avg_loss:.4f}")
    
    # Step 6: Evaluation
    print("\n5. Evaluating on test set...")
    predictions, true_labels = evaluate_model(model, test_loader, Config.DEVICE)
    
    # Get label names for reporting
    label_names = list(Config.INTENT_LABELS.keys())
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=label_names))
    
    # Step 7: Visualizations
    print("\n6. Generating visualizations...")
    plot_confusion_matrix(true_labels, predictions, label_names)
    plot_training_history(train_losses)
    
    # Step 8: Save model
    print("\n7. Saving model...")
    os.makedirs('saved_model', exist_ok=True)
    model.save_pretrained('saved_model')
    tokenizer.save_pretrained('saved_model')
    print("Model saved to 'saved_model/' directory")
    
    # Demo inference
    print("\n8. Demo Inference:")
    demo_questions = [
        "What medications are available for treating diabetes?",
        "How can I prevent heart disease?",
        "What are the symptoms of pneumonia?"
    ]
    
    model.eval()
    for question in demo_questions:
        encoding = tokenizer.encode_plus(
            question,
            add_special_tokens=True,
            max_length=Config.MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(Config.DEVICE)
            attention_mask = encoding['attention_mask'].to(Config.DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_idx = torch.argmax(outputs.logits, dim=1).item()
            
        predicted_intent = label_names[pred_idx]
        print(f"Q: {question}")
        print(f"Predicted Intent: {predicted_intent}\n")
    
    print("Training complete!")


if __name__ == "__main__":
    main()