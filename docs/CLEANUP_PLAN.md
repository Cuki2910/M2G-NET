# 🧹 Cleanup & Restructure Plan

**Date:** 2026-05-10  
**Goal:** Tối ưu cấu trúc folder, xóa files không cần thiết

---

## 📋 Files CÓ THỂ XÓA (Duplicates/Outdated)

### **Documentation (Outdated versions):**
- ❌ `comparison_v1_vs_v2.md` → Đã merge vào `tracking.md`
- ❌ `implementation_plan.md` → Version cũ, giữ `implementation_plan_v2.md`
- ❌ `tg_mvmt_gfnet_methodology.md` → Version cũ, giữ `tg_mvmt_gfnet_methodology_v2.md`

### **Scripts (One-time use):**
- ❌ `generate_synthetic_data.py` → Đã chạy xong, data đã có
- ❌ `generate_independent_test_set.py` → Đã chạy xong, test set đã có

**Tổng cộng: 5 files có thể xóa**

---

## 📁 CẤU TRÚC ĐỀ XUẤT (Optimized)

```
TG-MVMT-GFNet/
│
├── 📁 docs/                          # All documentation
│   ├── FINAL_SUMMARY.md              # ⭐ Main overview
│   ├── tracking.md                   # Progress tracking
│   ├── FIXES_SUMMARY.md              # Fixes applied
│   ├── INDEPENDENT_TEST_RESULTS.md   # Generalization results
│   ├── implementation_plan_v2.md     # Architecture design
│   ├── tg_mvmt_gfnet_methodology_v2.md  # Methodology
│   └── CLAUDE.md                     # AI instructions
│
├── 📁 src/                           # Core model code
│   ├── __init__.py
│   ├── data_pipeline.py
│   ├── views.py
│   ├── interaction.py
│   ├── fusion.py                     # ⭐ Fixed gated fusion
│   ├── loss.py
│   ├── model.py                      # ⭐ Main model
│   └── metrics.py
│
├── 📁 scripts/                       # Executable scripts
│   ├── train.py                      # ⭐ Training
│   ├── evaluate/
│   │   ├── ablation.py
│   │   ├── interpret.py
│   │   └── test_on_independent_set.py
│   ├── explain/
│   │   ├── visualize.py              # ⭐ 5 plots
│   │   ├── ig_explain.py             # ⭐ Integrated Gradients
│   │   └── case_study.py             # ⭐ Narrative explanations
│   └── search/
│       └── hparam_search.py          # Hyperparameter search
│
├── 📁 baselines/
│   └── run_all_baselines.py
│
├── 📁 data/
│   ├── raw/
│   │   └── synthetic_rider_data.csv
│   ├── processed/
│   │   └── (encoders, vocab)
│   └── test/
│       └── independent_test_set.csv
│
├── 📁 outputs/                       # Results
│   ├── case_studies.md               # ⭐ 7 case studies
│   ├── gate_heatmap.png
│   ├── per_sample_attention.png
│   ├── confusion_matrices.png
│   ├── ig_vs_gate.png
│   └── gate_weights.png
│
├── 📁 checkpoints/
│   └── best_model.pt                 # ⭐ Trained model
│
├── config.py                         # ⭐ Hyperparameters
├── requirements.txt
└── README.md (optional)
```

---

## 🎯 BENEFITS

### **Before (Current):**
- 20+ files ở root level
- Documentation lẫn lộn với code
- Scripts không có tổ chức
- Khó tìm file cần thiết

### **After (Proposed):**
- ✅ Root level: chỉ 3 items (config.py, requirements.txt, README.md)
- ✅ Documentation tập trung trong `docs/`
- ✅ Scripts phân loại theo chức năng
- ✅ Dễ navigate và review

---

## 📝 MIGRATION STEPS

### **Step 1: Create new folders**
```bash
mkdir -p docs
mkdir -p scripts/evaluate
mkdir -p scripts/explain
mkdir -p scripts/search
```

### **Step 2: Move documentation**
```bash
mv *.md docs/
mv docs/README.md .  # Keep README at root
```

### **Step 3: Move scripts**
```bash
mv train.py scripts/
mv ablation.py interpret.py test_on_independent_set.py scripts/evaluate/
mv visualize.py ig_explain.py case_study.py scripts/explain/
mv hparam_search.py scripts/search/
```

### **Step 4: Delete outdated files**
```bash
rm docs/comparison_v1_vs_v2.md
rm docs/implementation_plan.md
rm docs/tg_mvmt_gfnet_methodology.md
rm generate_synthetic_data.py
rm generate_independent_test_set.py
```

### **Step 5: Update imports**
- Update `train.py` imports: `import config` → `import sys; sys.path.insert(0, '..')`
- Update all scripts in `scripts/` to handle new paths

---

## ⚠️ RISKS & MITIGATION

### **Risk 1: Import errors**
- **Mitigation:** Add `sys.path.insert(0, '..')` to all scripts
- **Test:** Run each script after moving

### **Risk 2: Hardcoded paths**
- **Mitigation:** Use relative paths from project root
- **Check:** Search for hardcoded paths: `grep -r "a:\\\\GREEN-X" .`

### **Risk 3: Git history**
- **Mitigation:** Use `git mv` instead of `mv` (if using git)

---

## 🚀 EXECUTION

**Option A: Manual (Safe)**
1. Create backup: `cp -r . ../TG-MVMT-GFNet-backup`
2. Execute steps 1-5 manually
3. Test all scripts
4. Delete backup if successful

**Option B: Automated (Fast)**
1. Run migration script (to be created)
2. Test all scripts
3. Rollback if issues

---

## 📊 FINAL STRUCTURE SUMMARY

| Category | Before | After | Change |
|----------|--------|-------|--------|
| Root files | 20+ | 3 | -17 |
| Folders | 8 | 11 | +3 |
| Documentation | Scattered | `docs/` | Organized |
| Scripts | Root | `scripts/` | Categorized |
| Cleanliness | 3/10 | 9/10 | +6 |

---

## ✅ RECOMMENDATION

**Có nên cleanup không?**
- ✅ **YES** nếu: Bạn muốn dễ navigate, review, hoặc share với người khác
- ❌ **NO** nếu: Đang trong quá trình phát triển nhanh, chưa muốn refactor

**Khi nào nên cleanup?**
- ✅ Trước khi submit paper
- ✅ Trước khi share với advisor/collaborators
- ✅ Trước khi deploy production
- ✅ Khi có thời gian (30-60 phút)

**Độ ưu tiên:** Medium-High (không urgent nhưng nên làm sớm)
