import pandas as pd

# Read the CSV file
df = pd.read_csv('test_codes (2).csv')

print("Original data sample:")
print(df[['particle', 'particle_wt', 'particle2', 'particle_wt2']].head(20))

# Fix particle type when particle weight is 0
# For first layer
mask1 = df['particle_wt'] == 0
df.loc[mask1, 'particle'] = 'na'

# For second layer (if it exists)
if 'particle_wt2' in df.columns:
    mask2 = df['particle_wt2'] == 0
    df.loc[mask2, 'particle2'] = 'na'

print("\nAfter fixing:")
print(df[['particle', 'particle_wt', 'particle2', 'particle_wt2']].head(20))

# Save the corrected file
df.to_csv('test_codes (2).csv', index=False)

print(f"\nFixed {mask1.sum()} first layer entries and {mask2.sum() if 'particle_wt2' in df.columns else 0} second layer entries")
print("File saved as 'test_codes (2).csv'") 