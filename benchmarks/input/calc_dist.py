import pandas as pd

# Load the CSV
df = pd.read_csv('balanced_split_desc_train.csv')

print('='*60)
print('CLASS DISTRIBUTION ANALYSIS')
print('='*60)
print(f'\nTotal samples: {len(df)}')

print('\n' + '='*60)
print('DIAGNOSTIC CLASS DISTRIBUTION')
print('='*60)
diag_counts = df['diagnostic_class'].value_counts()
print(diag_counts)
print(f'\nPercentages:')
for cls, count in diag_counts.items():
    print(f'  {cls}: {count/len(df)*100:.2f}%')

print('\n' + '='*60)
print('SUBTYPE DISTRIBUTION')
print('='*60)
subtype_counts = df['subtype'].value_counts()
print(subtype_counts)
print(f'\nPercentages:')
for cls, count in subtype_counts.items():
    print(f'  {cls}: {count/len(df)*100:.2f}%')

print('\n' + '='*60)
print('ANATOMICAL SUBCLASS DISTRIBUTION')
print('='*60)
anatomical_counts = df['anatomical_subclass'].value_counts(dropna=False)
print(anatomical_counts)
print(f'\nPercentages:')
for cls, count in anatomical_counts.items():
    print(f'  {cls}: {count/len(df)*100:.2f}%')

print('\n' + '='*60)
print('CROSS-TABULATION: DIAGNOSTIC CLASS vs SUBTYPE')
print('='*60)
cross_tab = pd.crosstab(df['diagnostic_class'], df['subtype'], margins=True)
print(cross_tab)

print('\n' + '='*60)
print('CROSS-TABULATION: DIAGNOSTIC CLASS vs ANATOMICAL SUBCLASS')
print('='*60)
cross_tab2 = pd.crosstab(df['diagnostic_class'], df['anatomical_subclass'], margins=True, dropna=False)
print(cross_tab2)
