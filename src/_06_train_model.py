import torch
import torch.nn as nn
from _04_custom_dataset import get_dataloaders
from _05_model_definition import MultiTaskModel
# from transformers import AdamW
from torch.optim import AdamW
from _07_evaluate_model import evaluate_multitask
import os
from itertools import cycle

def train_multitask(model, emotion_train_loader, sarcasm_train_loader, emotion_val_loader, sarcasm_val_loader, device, save_dir, epochs=3, patience=2, lr=2e-5):
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_combined_f1 = 0.0
    patience_counter = 0
    best_model_state = None
    scaling_factor = 0.6

    for epoch in range(epochs):
        model.train()
        total_emotion_loss = 0.0
        total_sarcasm_loss = 0.0
        step = 1

        for emotion_batch, sarcasm_batch in zip(emotion_train_loader, cycle(sarcasm_train_loader)):
            optimizer.zero_grad()
            # Emotion forward
            print(f"Epoch {epoch+1}, Step {step}: Processing Emotion Task")
            emotion_input_ids = emotion_batch['input_ids'].to(device)
            emotion_attention_mask = emotion_batch['attention_mask'].to(device)
            emotion_labels = emotion_batch['labels'].to(device)

            # Sarcasm forward
            print(f"Epoch {epoch+1}, Step {step}: Processing Sarcasm Task")
            sarcasm_input_ids = sarcasm_batch['input_ids'].to(device)
            sarcasm_attention_mask = sarcasm_batch['attention_mask'].to(device)
            sarcasm_labels = sarcasm_batch['labels'].to(device)
            
            step+=1
            
            print("Getting the logits for both tasks...")
            emotion_logits= model(
                emotion_input_ids, emotion_attention_mask, task="emotion"
            )
            sarcasm_logits = model(
                sarcasm_input_ids, sarcasm_attention_mask, task="sarcasm"
            )

            # Calculate losses
            print("Calculating losses for both tasks...")
            emotion_loss = loss_fn(emotion_logits, emotion_labels)
            print("Emotion loss: ", emotion_loss.item())
            sarcasm_loss = loss_fn(sarcasm_logits, sarcasm_labels)
            print("Sarcasm loss: ", sarcasm_loss.item())
            loss = emotion_loss + sarcasm_loss * scaling_factor
            print("Total loss: ", loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # To prevent exploding gradients
            optimizer.step()

            total_emotion_loss += emotion_loss.item()
            total_sarcasm_loss += sarcasm_loss.item()

        avg_emotion_loss = total_emotion_loss / len(emotion_train_loader)
        avg_sarcasm_loss = total_sarcasm_loss / len(sarcasm_train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Emotion Loss: {avg_emotion_loss:.4f}, Sarcasm Loss: {avg_sarcasm_loss:.4f}")

        # Evaluate on validation set WITHOUT saving results files
        combined_macro_f1 = evaluate_multitask(
            model, emotion_val_loader, sarcasm_val_loader, device, save=False
        )
        print(f"Validation Combined Macro F1: {combined_macro_f1:.4f}")

        if combined_macro_f1 > best_combined_f1:
            best_combined_f1 = combined_macro_f1
            best_model_state = model.state_dict()
            patience_counter = 0
            print("New best model found. Saving model state...")

        else:
            patience_counter += 1
            print(f"No improvement. Patience counter: {patience_counter}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # After training finishes, load best model and save evaluation results only once
    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, os.path.join("results", "best_multitask_model.pth"))
        print("Best model saved to 'results/best_multitask_model.pth'")

        # Now evaluate and save results one time for the best model
        evaluate_multitask(
            model, emotion_val_loader, sarcasm_val_loader, device, save_dir, save=True
        )

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set paths and parameters
    emotion_batch_size = 64 # Emotion has 24816 rows of data
    sarcasm_batch_size = 16 # Sarcasm has 3681 rows of data only (almost 6.7 times less)
    # number of batches per epoch for emotion = 24816 / 64 = 387
    # number of batches per epoch for sarcasm = 3681 / 16 = 230
    epochs = 3
    patience = 2
    learning_rate = 2e-5
    train_val_results_dir = "./results/train_val"
    test_results_dir = "./results/test"

    os.makedirs(train_val_results_dir, exist_ok=True)
    os.makedirs(test_results_dir, exist_ok=True)

    # Prepare the data loaders (implement get_dataloaders in dataset.py)
    print("Loading dataloaders...")
    emotion_train_loader, sarcasm_train_loader, emotion_val_loader, sarcasm_val_loader, emotion_test_loader, sarcasm_test_loader = get_dataloaders(
        emotion_batch_size=emotion_batch_size,
        sarcasm_batch_size=sarcasm_batch_size
    )
    print("Dataloaders loaded.")

    # Initialize model
    print("Initializing model...")
    model = MultiTaskModel().to(device)
    print("Done initializing model.")

    # Train model
    print("Start training the model...")
    train_multitask(
        model,
        emotion_train_loader,
        sarcasm_train_loader,
        emotion_val_loader,
        sarcasm_val_loader,
        device,
        save_dir=train_val_results_dir,
        epochs=epochs,
        patience=patience,
        lr=learning_rate,
    )

if __name__ == "__main__":
    main()