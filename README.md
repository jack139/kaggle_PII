# Kaggle's PII Data Detection

[The Learning Agency Lab - PII Data Detection](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data)

Develop automated techniques to detect and remove PII from educational data.



## 转换数据

```bash
python3.9 convert_43k.py
python3.9 convert_43k_csv.py
python3.9 convert_10k.py
python3.9 convert_names.py
python3.9 convert.py
```



## 模型训练

```bash
python3.9 NER_gp.py
```



## 生成结果

```bash
echo "import NER_gp" | python3.9 
```
