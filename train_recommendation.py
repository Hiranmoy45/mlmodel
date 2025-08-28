import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.compose import ColumnTransformer
import numpy as np
import pickle
import warnings
from pydantic import BaseModel

# Suppress FutureWarning from pandas
warnings.simplefilter(action='ignore', category=FutureWarning)

# File paths
DATA_FILE = "../data/listings.csv"
PREPROCESSOR_FILE = "preprocessor.pkl"
PROCESSED_DF_FILE = "processed_df.pkl"
FEATURE_MATRIX_FILE = "feature_matrix.pkl"

# Placeholder Pydantic model for the incoming wishlist data
class WishlistItem(BaseModel):
    title: str
    description: str
    property_type: str
    address: str
    Location: str
    city: str
    state: str
    country: str
    postal_code: str
    price_per_night: float
    bedrooms: int
    bathrooms: int
    max_guests: int
    amenities: list
    total_area: int
    Price_per_SQFT: float

def preprocess_and_train():
    """Loads, preprocesses the data, and creates the combined feature matrix."""
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: The file '{DATA_FILE}' was not found. Please ensure it is in the correct directory.")
        return None, None, None

    # Handle missing values and data type conversions
    df.fillna({'Description': ''}, inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    if 'Property Type' not in df.columns:
        df['Property Type'] = 'Unknown'
    
    df.fillna('Unknown', inplace=True)

    def convert_price(price_str):
        if pd.isna(price_str) or not isinstance(price_str, str):
            return 0.0
        
        price_str = price_str.strip().replace('₹', '').replace(',', '')
        
        try:
            if 'Cr' in price_str:
                return float(price_str.replace('Cr', '').strip()) * 1e7
            elif 'L' in price_str:  # also handle Lakhs if needed
                return float(price_str.replace('L', '').strip()) * 1e5
            else:
                return float(price_str)
        except ValueError:
            return 0.0


    df['Price'] = df['Price'].astype(str).apply(convert_price)
    df['Price_per_SQFT'] = pd.to_numeric(df['Price_per_SQFT'], errors='coerce').fillna(0)
    df['Balcony'] = df['Balcony'].replace({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    df['Baths'] = pd.to_numeric(df['Baths'], errors='coerce').fillna(0).astype(int)
    
    # Correct column names to match your CSV file
    df = df.rename(columns={'Property T': 'Property Type', 'Price_per_SQFT ': 'Price_per_SQFT', 'Total_Area ': 'Total_Area', 'Baths ': 'Baths', 'Description ': 'Description'})

    # Define feature processing pipelines
    numeric_features = ['Price', 'Total_Area', 'Price_per_SQFT', 'Baths', 'Balcony']
    categorical_features = ['Location', 'Property Type']
    text_features = 'Description'

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('text', TfidfVectorizer(stop_words='english'), text_features)
        ]
    )
    
    feature_matrix = preprocessor.fit_transform(df)

    with open(PREPROCESSOR_FILE, 'wb') as f: pickle.dump(preprocessor, f)
    with open(PROCESSED_DF_FILE, 'wb') as f: pickle.dump(df, f)
    with open(FEATURE_MATRIX_FILE, 'wb') as f: pickle.dump(feature_matrix, f)

    print(f"Model trained and saved successfully! Feature matrix has {feature_matrix.shape[1]} features.")
    return df, preprocessor, feature_matrix

# --- Fixed get_recommendations function ---
def get_recommendations(wishlist_data, df, preprocessor, feature_matrix):
    """Generates recommendations based on the user's wishlist, with a flexible location filter."""
    if not wishlist_data:
        return []

    wishlist_list_of_dicts = [item.model_dump() for item in wishlist_data]
    wishlist_df = pd.DataFrame(wishlist_list_of_dicts)
    
    # Get all unique locations from the wishlist
    wishlist_locations = wishlist_df['Location'].apply(lambda x: x.lower()).unique().tolist()
    
    # Create a filter based on all locations in the wishlist
    location_mask = pd.Series([False] * len(df))
    for loc in wishlist_locations:
        keywords = [keyword.strip() for keyword in loc.split(',')]
        if keywords:
            # Use regex to find any of the keywords
            regex_pattern = '|'.join(keywords)
            location_mask = location_mask | df['Location'].str.lower().str.contains(regex_pattern, na=False)

    # Filter the entire dataset and feature matrix based on the location keywords
    if location_mask.any():
        filtered_df = df[location_mask].copy()
        filtered_indices = filtered_df.index
        filtered_feature_matrix = feature_matrix[filtered_indices]
    else:
        # If no matching location is found, return an empty list
        print("No properties found for the specified locations.")
        return []

    # Map incoming JSON keys to the column names used by the preprocessor
    rename_dict = {
        'description': 'Description',
        'property_type': 'Property Type',
        'Location': 'Location',
        'price_per_night': 'Price',
        'total_area': 'Total_Area',
        'bathrooms': 'Baths',
    }
    wishlist_df = wishlist_df.rename(columns=rename_dict)
    
    for col in preprocessor.feature_names_in_:
        if col not in wishlist_df.columns:
            if df[col].dtype in ['float64', 'int64']:
                wishlist_df[col] = 0.0
            else:
                wishlist_df[col] = 'Unknown'
    
    if 'amenities' in wishlist_df.columns and not wishlist_df['amenities'].empty:
        wishlist_df['Balcony'] = wishlist_df['amenities'].apply(lambda x: 1 if "Balcony" in str(x) else 0)
    else:
        wishlist_df['Balcony'] = 0

    wishlist_matrix = preprocessor.transform(wishlist_df[preprocessor.feature_names_in_])
    
    # Calculate similarity with the filtered feature matrix
    cosine_similarities = cosine_similarity(wishlist_matrix, filtered_feature_matrix)
    summed_scores = np.sum(cosine_similarities, axis=0)
    top_indices = np.argsort(summed_scores)[::-1]
    
    wishlist_titles = {item.title for item in wishlist_data}
    recommended_indices = []
    
    for idx in top_indices:
        original_idx = filtered_df.iloc[idx].name
        property_title = df.loc[original_idx, 'Property Title']
        if property_title not in wishlist_titles:
            recommended_indices.append(original_idx)
        if len(recommended_indices) >= 10:
            break
            
    return df.loc[recommended_indices].to_dict('records')

if __name__ == "__main__":
    df, preprocessor, feature_matrix = preprocess_and_train()
    if df is not None:
        print("Training complete. The API is ready to be run.")

        # Example with multiple wishlist items, including different locations
        test_wishlist_data = [
            WishlistItem(
                title="Luxury Villa",
                description="Sea-facing villa with an infinity pool and modern amenities.",
                property_type="Villa",
                address="Bengalore",
                Location="chennai",  
                city="Bangalore",
                state="kolkata",
                country="India",
                postal_code="560100",
                price_per_night=1100,
                bedrooms=4,
                bathrooms=4,
                max_guests=8,
                amenities=["Pool", "WiFi", "Parking", "Balcony"],
                total_area=2590,
                Price_per_SQFT=1100/2590
            ),
            WishlistItem(
                title="City Apartment",
                description="Modern apartment in the heart of the city, close to transport.",
                property_type="Apartment",
                address="New Delhi",
                Location="Delhi",
                city="New Delhi",
                state="Delhi",
                country="India",
                postal_code="110001",
                price_per_night=600,
                bedrooms=2,
                bathrooms=2,
                max_guests=4,
                amenities=["WiFi", "Parking", "Gym"],
                total_area=1200,
                Price_per_SQFT=600/1200
            )
        ]
        
        recommendations = get_recommendations(test_wishlist_data, df, preprocessor, feature_matrix)
        
        if recommendations:
            print("\nRecommendations based on your wishlist (filtered by all wishlist locations):")
            for rec in recommendations:
                print(f"- {rec.get('Property Title', 'N/A')} | "
                      f"Type: {rec.get('Property Type', 'N/A')} | "
                      f"Location: {rec.get('Location', 'N/A')} | "
                      f"Price: {rec.get('Price', 'N/A')} | "
                      f"Price_per_SQFT: {rec.get('Price_per_SQFT', 'N/A')} | "
                      f"Area: {rec.get('Total_Area', 'N/A')} | "
                      f"Baths: {rec.get('Baths', 'N/A')}")
        else:
            print("\nNo recommendations found.")