import os
import torch

def save_checkpoints(encoder, decoder, epoch, output_dir="checkpoints"):
    """
    Saves the state dictionaries of the encoder and decoder models separately.
    """
    os.makedirs(output_dir, exist_ok=True)
    encoder_path = os.path.join(output_dir, f"encoder_epoch_{epoch}.pth")
    decoder_path = os.path.join(output_dir, f"decoder_epoch_{epoch}.pth")
    
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)
    print(f"Saved checkpoints at epoch {epoch}: {encoder_path}, {decoder_path}")
