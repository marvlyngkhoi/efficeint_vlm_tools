import torch
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display
import cv2

def plot_attention_head(attention_tensor, head_index, title_prefix=""):
    """
    Plots the attention map of a specific head from an attention tensor.
    Args:
        attention_tensor (torch.Tensor): The attention tensor of shape [batch_size, num_heads, seq_len, seq_len]
        head_index (int): The index of the head to plot
        title_prefix (str): Optional prefix for the plot title
    """
    # Ensure the attention tensor has the correct shape
    if len(attention_tensor.shape) != 4:
        raise ValueError("The attention tensor must have 4 dimensions: [batch_size, num_heads, seq_len, seq_len]")
    
    # Get the batch size, number of heads, and sequence length
    batch_size, num_heads, seq_len, _ = attention_tensor.shape
    
    # Check if the head_index is valid
    if head_index >= num_heads:
        raise ValueError(f"head_index must be less than the number of heads ({num_heads})")
    
    # Select the attention map for the specified head (for the first item in the batch) 
    attention_map = attention_tensor[0, head_index]
    
    # Convert the attention map to float32 if it's in an unsupported dtype
    if attention_map.dtype == torch.bfloat16:
        attention_map = attention_map.float()
    
    # Detach from computation graph and move to CPU for plotting
    # attention_map = attention_map.detach().cpu().numpy()
    
    # # Create the plot
    # plt.figure(figsize=(20, 20))
    # im = plt.imshow(attention_map, cmap='viridis', vmin=0, vmax=1)
    # plt.colorbar(im, label='Attention Weight')
    # plt.title(f'{title_prefix}Attention Map for Head {head_index}')
    # plt.xlabel('Key Positions')
    # plt.ylabel('Query Positions')
    # plt.show()
    
    attention_map = attention_map.detach().cpu().numpy()

    # Interpolate (Extrapolate) using bicubic interpolation
    resized_map = cv2.resize(attention_map, (256, 256), interpolation=cv2.INTER_CUBIC)

    # Normalize the values for better contrast
    normalized_map = (resized_map - resized_map.min()) / (resized_map.max() - resized_map.min() + 1e-8)

    # Create the plot
    plt.figure(figsize=(10, 10))
    im = plt.imshow(normalized_map, cmap='magma', vmin=0, vmax=1)
    plt.colorbar(im, label='Attention Weight')
    plt.title(f'{title_prefix} Enhanced Attention Map for Head {head_index}')
    plt.colorbar(im)
    plt.xlabel('Key Positions')
    plt.ylabel('Query Positions')
    plt.show()

class AttentionVisualizer:
    def __init__(self, attention_maps):
        self.attention_maps = attention_maps
        
        # Get available layers
        self.text_layers = sorted([k for k in attention_maps.keys() if k.startswith('text_model.layers') and k.endswith('self_attn')])
        self.vision_layers = sorted([k for k in attention_maps.keys() if k.startswith('vision_model.encoder.layers') and k.endswith('self_attn')])
        
        # Create dropdown for selecting between text and vision attention
        self.model_dropdown = widgets.Dropdown(
            options=['Text Attention', 'Vision Attention'],
            description='Model:',
            style={'description_width': 'initial'}
        )
        
        # Create dropdown for layer selection
        self.layer_dropdown = widgets.Dropdown(
            options=self.text_layers,  # Initial options are text layers
            description='Layer:',
            style={'description_width': 'initial'}
        )
        
        # Create slider for selecting head index
        self.head_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=self._get_max_heads(self.text_layers[0]) - 1,
            description='Head Index:',
            style={'description_width': 'initial'}
        )
        
        # For vision model, create patch selector
        self.patch_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=self._get_max_patches(self.vision_layers[0]) - 1,
            description='Patch Index:',
            style={'description_width': 'initial'}
        )
        
        # Create output widget for the plot
        self.output = widgets.Output()
        
        # Set up observers
        self.model_dropdown.observe(self.on_model_change, names='value')
        self.layer_dropdown.observe(self.on_layer_change, names='value')
        self.head_slider.observe(self.on_head_change, names='value')
        self.patch_slider.observe(self.on_patch_change, names='value')
        
        # Initial update
        self._update_plot()
    
    def _get_max_heads(self, layer_key):
        return self.attention_maps[layer_key].shape[1]
    
    def _get_max_patches(self, layer_key):
        return len(self.attention_maps[layer_key])
    
    def on_model_change(self, change):
        # Update layer options based on model selection
        if change['new'] == 'Text Attention':
            self.layer_dropdown.options = self.text_layers
            self.patch_slider.layout.visibility = 'hidden'
        else:
            self.layer_dropdown.options = self.vision_layers
            self.patch_slider.layout.visibility = 'visible'
        
        # Update head slider range based on new layer
        self.head_slider.max = self._get_max_heads(self.layer_dropdown.value) - 1
        
        # Update patch slider range for vision model
        if change['new'] == 'Vision Attention':
            self.patch_slider.max = self._get_max_patches(self.layer_dropdown.value) - 1
        
        self._update_plot()
    
    def on_layer_change(self, change):
        # Update head slider range
        self.head_slider.max = self._get_max_heads(change['new']) - 1
        
        # Update patch slider range for vision model
        if self.model_dropdown.value == 'Vision Attention':
            self.patch_slider.max = self._get_max_patches(change['new']) - 1
        
        self._update_plot()
    
    def on_head_change(self, _):
        self._update_plot()
    
    def on_patch_change(self, _):
        if self.model_dropdown.value == 'Vision Attention':
            self._update_plot()
    
    def _update_plot(self):
        with self.output:
            self.output.clear_output(wait=True)
            
            layer_key = self.layer_dropdown.value
            if self.model_dropdown.value == 'Text Attention':
                attn_tensor = self.attention_maps[layer_key]
                title_prefix = f'Text Model (Layer {layer_key.split(".")[2]}): '
            else:
                vision_attn = self.attention_maps[layer_key]
                attn_tensor = vision_attn[self.patch_slider.value].unsqueeze(dim=0)
                title_prefix = f'Vision Model (Layer {layer_key.split(".")[3]}, Patch {self.patch_slider.value}): '
            
            plot_attention_head(attn_tensor, self.head_slider.value, title_prefix)
    
    def display(self):
        # Create layout
        controls = widgets.VBox([
            self.model_dropdown,
            self.layer_dropdown,
            self.head_slider,
            self.patch_slider
        ])
        
        # Initialize patch slider visibility
        self.patch_slider.layout.visibility = 'hidden'
        
        # Display widgets and output
        display(widgets.VBox([controls, self.output]))

# Example usage:
# visualizer = AttentionVisualizer(attention_maps)
# visualizer.display()
