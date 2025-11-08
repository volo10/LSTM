# Google Colab GPU Training Instructions

## Why Use Google Colab?

- **Free GPU**: Get access to NVIDIA T4 or better GPU for free
- **Fast Training**: Train in ~5-10 minutes on GPU vs ~30-60 minutes on CPU
- **No Setup**: Everything runs in the cloud, no local installation needed

## Step-by-Step Guide

### 1. Upload Notebook to Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click `File` → `Upload notebook`
3. Upload `train_colab.ipynb` from your LSTM2 folder

**OR** 

1. Open Google Drive
2. Upload `train_colab.ipynb` to your Drive
3. Right-click → `Open with` → `Google Colaboratory`

### 2. Enable GPU

**IMPORTANT:** You must enable GPU!

1. Click `Runtime` → `Change runtime type`
2. Under `Hardware accelerator`, select **GPU**
3. Click `Save`

### 3. Run Training

1. Click `Runtime` → `Run all` (or press `Ctrl+F9` / `Cmd+F9`)
2. Wait for all cells to execute (about 5-10 minutes with GPU)
3. Monitor the training progress in the output

### 4. Download Trained Model

After training completes:

1. The last cell will automatically trigger download of `best_model.pth`
2. Save it to your computer

### 5. Use Model Locally

Transfer the model to your Mac:

```bash
# Navigate to your project
cd /Users/bvolovelsky/Desktop/LLM/LSTM2

# Place the downloaded file in outputs/models/
mv ~/Downloads/best_model.pth outputs/models/

# Run evaluation
python3 evaluate.py

# Generate plots
python3 plot_results.py
```

## Expected Output

### During Training

You should see:
- GPU detected (e.g., "Tesla T4")
- Dataset creation: 40,000 training samples, 40,000 test samples
- Progress bars for each epoch
- Train and test loss decreasing
- Best model saved messages

### Training Time

- **With GPU (T4)**: ~5-10 minutes
- **With GPU (V100/A100)**: ~2-5 minutes
- **Without GPU (CPU)**: ~30-60 minutes

### Final Results

Expected metrics after training:
- MSE_train: ~0.01-0.05
- MSE_test: ~0.01-0.05
- Ratio (test/train): ~0.8-1.2 (good generalization)

## Troubleshooting

### "No GPU detected"

**Solution:** 
1. Go to `Runtime` → `Change runtime type`
2. Select `GPU` under Hardware accelerator
3. Click `Save`
4. Re-run all cells

### "Quota exceeded"

Google Colab has usage limits on free GPUs.

**Solutions:**
- Wait a few hours and try again
- Use Colab Pro ($9.99/month) for more GPU time
- Train on your local CPU (slower but works)

### "Out of memory"

The model is small and shouldn't cause this, but if it happens:

**Solution:**
- Restart runtime: `Runtime` → `Restart runtime`
- Try again

### Downloads don't work

**Solution:**
```python
# Alternative download method (add this cell at the end)
!zip model.zip best_model.pth training_curves.png
from google.colab import files
files.download('model.zip')
```

## Tips

### Save to Google Drive (Optional)

Add this cell after training to save to your Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

!cp best_model.pth /content/drive/MyDrive/
print("Model saved to Google Drive!")
```

### Monitor GPU Usage

Add this cell to check GPU utilization:

```python
!nvidia-smi
```

### Run Multiple Times

If you want to train with different hyperparameters:

1. Modify the configuration in the "Train the Model" cell
2. Run just that cell again
3. Compare results

## What Gets Downloaded

After training, you'll have:

1. **best_model.pth** (~223 KB)
   - Contains model weights
   - Optimizer state
   - Training metrics
   
2. **training_curves.png** (optional)
   - Visualization of training progress

## Next Steps After Download

Once you have the model on your Mac:

```bash
# 1. Verify model is in place
ls -lh outputs/models/best_model.pth

# 2. Evaluate the model
python3 evaluate.py

# 3. Generate all visualizations
python3 plot_results.py

# 4. Check results
open outputs/plots/signal_reconstruction.png
```

You should see:
- `outputs/results/evaluation_results.json` - Metrics
- `outputs/results/reconstructed_signals.npz` - Signal data
- `outputs/plots/signal_reconstruction.png` - Visualizations
- `outputs/plots/per_frequency_mse.png` - Performance by frequency

## Advantages of This Approach

✓ **Fast**: GPU training is 3-6× faster than CPU
✓ **Free**: No cost for GPU access (with limits)
✓ **Easy**: No local setup required
✓ **Reproducible**: Same seeds give same results
✓ **Portable**: Train anywhere, use model locally

## Cost Comparison

| Option | Time | Cost |
|--------|------|------|
| Local CPU (Mac) | 30-60 min | $0 |
| Colab Free (GPU) | 5-10 min | $0 |
| Colab Pro (GPU) | 3-5 min | $9.99/month |
| AWS p3.2xlarge | 3-5 min | ~$3/hour |

**Recommendation**: Use Colab Free for this project!

## Questions?

If something doesn't work:
1. Check the troubleshooting section above
2. Verify GPU is enabled
3. Check you have ~40,000 samples (not 400M!)
4. Look at the error messages carefully

Happy training! 🚀

