from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("/etc/secrets/sparexchange-323c8-firebase-adminsdk-fbsvc-dfed029bc9.json")
firebase_admin.initialize_app(cred)
# Get Firestore client
db = firestore.client()


app = Flask(__name__)
def recommend_products(products, transactions, top_n=5):
    df_products = pd.DataFrame(products)
    df_tx = pd.DataFrame(transactions)

    # Step 1: Filter purchased items
    purchased_ids = df_tx['itemId'].unique()
    purchased_df = df_products[df_products['id'].isin(purchased_ids)].copy()

    # Step 2: Combine relevant text fields
    def combine_text_fields(row):
        return f"{row['name']} {row['description']} {row['category']} {row['ownerId']}"

    df_products['text'] = df_products.apply(combine_text_fields, axis=1)
    purchased_df['text'] = purchased_df.apply(combine_text_fields, axis=1)

    # Step 3: TF-IDF
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df_products['text'])

    # Step 4: Convert timestamps
    df_products['createdAt'] = pd.to_datetime(df_products['createdAt'])
    df_products['updatedAt'] = pd.to_datetime(df_products['updatedAt'])
    df_products['createdAt_ts'] = df_products['createdAt'].astype(np.int64) // 10**9
    df_products['updatedAt_ts'] = df_products['updatedAt'].astype(np.int64) // 10**9

    # Step 5: Normalize numeric features
    scaler = MinMaxScaler()
    numeric_features = scaler.fit_transform(df_products[['price', 'createdAt_ts', 'updatedAt_ts']])

    # Step 6: Combine all features (text + numbers)
    full_features = hstack([tfidf_matrix, numeric_features]).tocsr()

    # Step 7: Compute average vector for purchased items
    purchased_indices = df_products[df_products['id'].isin(purchased_ids)].index
    purchased_vectors = full_features[purchased_indices]
    user_profile = np.asarray(purchased_vectors.mean(axis=0)).ravel()

    # Step 8: Cosine similarity
    similarities = cosine_similarity(user_profile.reshape(1, -1), full_features).flatten()


    # Step 9: Filter and sort
    df_products['similarity'] = similarities
    recommendations = df_products[~df_products['id'].isin(purchased_ids)]
    top_recommendations = recommendations.sort_values(by='similarity', ascending=False).head(top_n)

    return top_recommendations[['id', 'name', 'category', 'price', 'description']].to_dict(orient='records')

# ===============================
# Example usage
# ===============================



# # 🔍 Recommend
# recommendations = recommend_products(products, transactions, top_n=3)

# # 🖨️ Print results
# print("Top product recommendations:")
# for i, product in enumerate(recommendations, 1):
#     print(f"{i}. {product['id']} {product['name']} ({product['category']}) - {product['price']} DT")
def fetch_products_from_firebase():
    products_ref = db.collection('Products')
    docs = products_ref.stream()

    products = []
    for doc in docs:
        product = doc.to_dict()
        product['id'] = doc.id
        products.append(product)
    print(products)
    return products

def fetch_transactions_for_user_from_firebase(user_id):
    transactions_ref = db.collection('Transactions').where('buyerId', '==', user_id)
    docs = transactions_ref.stream()

    transactions = []
    for doc in docs:
        transaction = doc.to_dict()
        transaction['id'] = doc.id
        transactions.append(transaction)
    print(transactions)
    return transactions

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "user_id query parameter is required"}), 400

    products = fetch_products_from_firebase()
    transactions = fetch_transactions_for_user_from_firebase(user_id)

    if not transactions:
        # If user has no transactions, return top products or empty
        return jsonify({"recommendations": []})

    top_n = int(request.args.get('top_n', 5))

    recommendations = recommend_products(products, transactions, top_n=top_n)

    return jsonify({
        "user_id": user_id,
        "recommendations": recommendations,
        "products": products,
        "transactions": transactions
    })


if __name__ == '__main__':
    app.run(debug=True)
