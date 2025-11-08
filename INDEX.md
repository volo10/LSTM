# LSTM Frequency Extraction - Project Index

Welcome! This index will help you navigate the project and understand what each file does.

## 🚀 Start Here

**New to this project?** Read these in order:

1. **[QUICKSTART.md](QUICKSTART.md)** - Get up and running in minutes
2. **[README.md](README.md)** - Complete documentation
3. **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Technical overview

## 📋 Documentation Files

### Getting Started
- **[QUICKSTART.md](QUICKSTART.md)** - Installation and quick start guide
- **[README.md](README.md)** - Comprehensive documentation with usage examples
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Technical summary and architecture overview

### Planning & Requirements
- **[planning.md](planning.md)** - High-level project plan, phases, and milestones
- **[tasks.md](tasks.md)** - Detailed task breakdown and checklist
- **[prd.md](prd.md)** - Product Requirements Document (inputs, outputs, metrics)
- **[claude.md](claude.md)** - AI assistant project definition and development workflow

### Assignment
- **[L2-homework.pdf](L2-homework.pdf)** - Original homework assignment

## 💻 Code Files

### Main Scripts (Run These)
- **[train.py](train.py)** - Train the LSTM model
- **[evaluate.py](evaluate.py)** - Evaluate trained model performance
- **[plot_results.py](plot_results.py)** - Generate visualizations
- **[test_system.py](test_system.py)** - Quick system verification test

### Core Modules (Implementation)
- **[model.py](model.py)** - LSTM model architecture
- **[data_generator.py](data_generator.py)** - Dataset generation and signal synthesis

## ⚙️ Configuration

- **[config.yaml](config.yaml)** - Hyperparameters and settings
- **[requirements.txt](requirements.txt)** - Python package dependencies

## 📊 Outputs (Generated During Use)

The following directories are created when you run the scripts:

```
outputs/
├── models/          # Saved model checkpoints
├── logs/            # Training history (JSON)
├── results/         # Evaluation results (JSON, NPZ)
└── plots/           # Visualizations (PNG)
```

## 🎯 Quick Reference by Task

### "I want to understand the project"
→ Read: **QUICKSTART.md** → **PROJECT_SUMMARY.md** → **README.md**

### "I want to see what it does"
→ Run: `python3 test_system.py`

### "I want to train a model"
→ Run: `python3 train.py`

### "I want to evaluate results"
→ Run: `python3 evaluate.py` then `python3 plot_results.py`

### "I want to modify hyperparameters"
→ Edit: **config.yaml**

### "I want to understand the architecture"
→ Read: **PROJECT_SUMMARY.md** (Section: Architecture)
→ Code: **model.py**

### "I want to understand the data"
→ Read: **prd.md** (Section: Signal Specifications)
→ Code: **data_generator.py**

### "I want to understand state management"
→ Read: **PROJECT_SUMMARY.md** (Section: State Management Strategy)
→ Code: **train.py** (lines 74-82)

### "I want to see the requirements"
→ Read: **prd.md**

### "I want implementation details"
→ Read code files with extensive docstrings and comments

## 📖 Documentation Guide

### For Beginners
Start with these, they're written to be accessible:
1. QUICKSTART.md
2. README.md (sections: Overview, Usage, Quick Start)
3. PROJECT_SUMMARY.md (sections: Overview, Architecture)

### For Implementers
If you want to understand or modify the code:
1. claude.md (development workflow)
2. planning.md (approach and phases)
3. Code files (model.py, data_generator.py, etc.)

### For Researchers
If you want deep technical understanding:
1. prd.md (complete specifications)
2. PROJECT_SUMMARY.md (technical details)
3. planning.md (methodology)
4. Code implementation

## 🔍 Find by Topic

### State Management
- **Concept**: PROJECT_SUMMARY.md → State Management Strategy
- **Implementation**: train.py (lines 74-82), evaluate.py
- **Why it matters**: claude.md → Implementation Guidelines

### Signal Processing
- **Theory**: prd.md → Signal Specifications
- **Implementation**: data_generator.py
- **Visualization**: Run plot_results.py

### Training Process
- **Overview**: planning.md → Phase 3
- **Configuration**: config.yaml
- **Implementation**: train.py
- **Usage**: QUICKSTART.md → Full Pipeline

### Model Architecture
- **Design**: PROJECT_SUMMARY.md → Architecture
- **Specification**: prd.md → Model Architecture
- **Implementation**: model.py

### Evaluation
- **Metrics**: prd.md → Evaluation Metrics
- **Implementation**: evaluate.py
- **Visualization**: plot_results.py

## 💡 Common Questions

**Q: Where do I start?**
A: Run `python3 test_system.py` to verify everything works, then read QUICKSTART.md

**Q: How do I train a model?**
A: `python3 train.py` - it will take 10-30 minutes

**Q: What's the key innovation here?**
A: Proper state management - maintaining LSTM state within signal instances while resetting between instances. See PROJECT_SUMMARY.md.

**Q: How do I change hyperparameters?**
A: Edit config.yaml and run train.py again

**Q: Where are the results saved?**
A: In outputs/ directory (models, logs, results, plots subdirectories)

**Q: How do I know if training succeeded?**
A: Check that Test MSE / Train MSE ratio is between 0.8 and 1.2

**Q: Can I use this for other frequencies?**
A: Yes! Edit config.yaml to change the frequency list

**Q: Why is batch size 1?**
A: To maintain proper temporal ordering for state management

## 🛠️ File Dependencies

```
train.py
├── requires: data_generator.py, model.py
└── outputs: outputs/models/, outputs/logs/

evaluate.py
├── requires: data_generator.py, model.py
├── requires: outputs/models/best_model.pth
└── outputs: outputs/results/

plot_results.py
├── requires: outputs/logs/, outputs/results/
└── outputs: outputs/plots/

test_system.py
├── requires: data_generator.py, model.py
└── outputs: terminal output only
```

## 📏 Project Statistics

- **Total Python Files**: 6 (train, evaluate, plot, model, data_generator, test)
- **Documentation Files**: 8 (md files)
- **Configuration Files**: 2 (config.yaml, requirements.txt)
- **Total Lines of Code**: ~2,000
- **Model Parameters**: ~18,000
- **Training Dataset Size**: 40M samples (1000 instances × 10s × 1000Hz × 4 frequencies)
- **Test Dataset Size**: 8M samples

## 🎓 Learning Path

### Level 1: User
1. Run test_system.py
2. Read QUICKSTART.md
3. Run the full pipeline (train, evaluate, plot)
4. Experiment with config.yaml

### Level 2: Developer
1. Read README.md thoroughly
2. Read PROJECT_SUMMARY.md
3. Study model.py and data_generator.py
4. Modify and experiment with the code

### Level 3: Researcher
1. Read all documentation files
2. Study the state management implementation
3. Analyze training dynamics
4. Explore extensions and modifications

## 📞 Support

If you're stuck:
1. Check QUICKSTART.md troubleshooting section
2. Run test_system.py to isolate the issue
3. Verify all files are present: `ls -la`
4. Check Python version: `python3 --version` (need 3.7+)

## ✅ Project Checklist

Use this to verify your setup:

- [ ] All Python files present (6 files)
- [ ] All documentation present (8 files)
- [ ] Dependencies installed (`pip3 install -r requirements.txt`)
- [ ] test_system.py runs successfully
- [ ] outputs/ directory structure created
- [ ] Ready to train!

---

**Last Updated**: November 8, 2025

**Version**: 1.0

**Status**: Complete ✓

Enjoy exploring LSTM-based frequency extraction! 🎵

