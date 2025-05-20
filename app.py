from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore, auth
from flask_cors import CORS

# Initialize Firebase
try:
    cred = credentials.Certificate("sparexchange-323c8-firebase-adminsdk-fbsvc-dfed029bc9.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    raise Exception(f"Failed to initialize Firebase: {e}")

app = Flask(__name__)
CORS(app, origins=["https://sparexchange.vercel.app"])

def recommend_products(products, transactions, top_n=5):
    try:
        df_products = pd.DataFrame(products)
        df_tx = pd.DataFrame(transactions)

        purchased_ids = df_tx['itemId'].unique()
        purchased_df = df_products[df_products['id'].isin(purchased_ids)].copy()

        def combine_text_fields(row):
            return f"{row['name']} {row['description']} {row['category']} {row['ownerId']}"

        df_products['text'] = df_products.apply(combine_text_fields, axis=1)
        purchased_df['text'] = purchased_df.apply(combine_text_fields, axis=1)

        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(df_products['text'])

        df_products['createdAt'] = pd.to_datetime(df_products['createdAt'])
        df_products['updatedAt'] = pd.to_datetime(df_products['updatedAt'])
        df_products['createdAt_ts'] = df_products['createdAt'].astype(np.int64) // 10**9
        df_products['updatedAt_ts'] = df_products['updatedAt'].astype(np.int64) // 10**9

        scaler = MinMaxScaler()
        numeric_features = scaler.fit_transform(df_products[['price', 'createdAt_ts', 'updatedAt_ts']])

        full_features = hstack([tfidf_matrix, numeric_features]).tocsr()

        purchased_indices = df_products[df_products['id'].isin(purchased_ids)].index
        purchased_vectors = full_features[purchased_indices]
        user_profile = np.asarray(purchased_vectors.mean(axis=0)).ravel()

        similarities = cosine_similarity(user_profile.reshape(1, -1), full_features).flatten()

        df_products['similarity'] = similarities
        recommendations = df_products[~df_products['id'].isin(purchased_ids)]
        top_recommendations = recommendations.sort_values(by='similarity', ascending=False).head(top_n)

        return top_recommendations[['id', 'name', 'category', 'price', 'description']].to_dict(orient='records')
    except Exception as e:
        raise Exception(f"Error in recommendation logic: {e}")

def fetch_products_from_firebase():
    try:
        products_ref = db.collection('Products')
        docs = products_ref.get()
        products = []
        for doc in docs:
            product = doc.to_dict()
            product['id'] = doc.id
            products.append(product)
        return products
    except Exception as e:
        raise Exception(f"Error fetching products from Firebase: {e}")

def fetch_transactions_for_user_from_firebase(user_id):
    try:
        transactions_ref = db.collection('Transactions').where('buyerId', '==', user_id)
        docs = transactions_ref.get()
        transactions = []
        for doc in docs:
            transaction = doc.to_dict()
            transaction['id'] = doc.id
            transactions.append(transaction)
        return transactions
    except Exception as e:
        raise Exception(f"Error fetching transactions for user {user_id}: {e}")

# Utility: Standardized response format
def format_response(status, data=None, error=None):
    return jsonify({
        "status": status,
        "data": data,
        "error": error if error else []
    })

# Routes

@app.route('/')
def api_documentation():
    docs = {
        "Welcome": "SpareXchange API - User Recommendation & Management System",
        "Routes": {
            "/recommend?user_id=<user_id>&top_n=<n>": "Get product recommendations for a user.",
            "/is_blocked?id=<user_id>": "Check if a user is blocked.",
            "/block?id=<user_id>": "Block a user.",
            "/unblock?id=<user_id>": "Unblock a user.",
            "/delete?id=<user_id>": "Delete a user."
        }
    }
    return format_response(True, data=docs)

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    top_n = request.args.get('top_n', 5)

    if not user_id:
        return format_response(False, error=["Missing required query parameter: user_id"]), 400
    try:
        top_n = int(top_n)
    except ValueError:
        return format_response(False, error=["Invalid value for top_n. Must be an integer."]), 400

    try:
        products = fetch_products_from_firebase()
        transactions = fetch_transactions_for_user_from_firebase(user_id)

        if not transactions:
            return format_response(True, data={
                "user_id": user_id,
                "recommendations": [],
                "message": "No transactions found for this user."
            })

        recommendations = recommend_products(products, transactions, top_n=top_n)

        return format_response(True, data={
            "user_id": user_id,
            "recommendations": recommendations,
            "products": products,
            "transactions": transactions
        })
    except Exception as e:
        return format_response(False, error=[str(e)]), 500

@app.route('/is_blocked', methods=['GET'])
def check_user_block_status():
    user_id = request.args.get('id')
    if not user_id:
        return format_response(False, error=["Missing user ID"]), 400
    try:
        user = auth.get_user(user_id)
        return format_response(True, data={
            "uid": user.uid,
            "is_blocked": user.disabled
        })
    except Exception as e:
        return format_response(False, error=[str(e)]), 500

@app.route('/block', methods=['POST'])
def block_user():
    user_id = request.args.get('id')
    if not user_id:
        return format_response(False, error=["Missing user ID"]), 400
    try:
        auth.update_user(user_id, disabled=True)
        return format_response(True, data={
            "uid": user_id,
            "is_blocked": True
        })
    except Exception as e:
        return format_response(False, error=[str(e)]), 500

@app.route('/unblock', methods=['POST'])
def unblock_user():
    user_id = request.args.get('id')
    if not user_id:
        return format_response(False, error=["Missing user ID"]), 400
    try:
        auth.update_user(user_id, disabled=False)
        return format_response(True, data={
            "uid": user_id,
            "is_blocked": False
        })
    except Exception as e:
        return format_response(False, error=[str(e)]), 500

@app.route('/delete', methods=['DELETE'])
def delete_user():
    user_id = request.args.get('id')
    if not user_id:
        return format_response(False, error=["Missing user ID"]), 400
    try:
        auth.delete_user(user_id)
        db.collection('Users').document(user_id).delete()
        return format_response(True, data={"message": f"User {user_id} deleted from Firebase Auth and Firestore."})
    except Exception as e:
        return format_response(False, error=[str(e)]), 500

if __name__ == '__main__':
    app.run(debug=True)
