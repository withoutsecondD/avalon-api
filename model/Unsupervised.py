import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import Isomap, TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import json
import warnings
warnings.filterwarnings('ignore')


# loading and preprocessing
def load_prepare_data(path):
    df = pd.read_csv(path)
    df = df[:10000]
    df = df.drop(columns=['1st_Road_Class', '2nd_Road_Class', 'Carriageway_Hazards', 'Light_Conditions', 'Special_Conditions_at_Site'])
    df.dropna(inplace=True)

    X = df.drop(columns=['Accident_Severity'])
    y = df['Accident_Severity']

    if y.dtype == object or y.dtype.name == 'category':
        y = LabelEncoder().fit_transform(y)

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    X = preprocessor.fit_transform(X)

    return X, y


# decomposition
def apply_decomposition(X, method='pca', n_components=5):
    if method == 'pca':
        model = PCA(n_components=n_components)
        X_red = model.fit_transform(X)
        return X_red, {
            'explained_variance_ratio': model.explained_variance_ratio_.tolist(),
            'noise_variance': getattr(model, 'noise_variance_', 0),
            'n_components': model.n_components_
        }

    elif method == 'nmf':
        model = NMF(n_components=n_components, init='random', random_state=0)
        scaler = MinMaxScaler()
        X_nonneg = scaler.fit_transform(X)  
        X_red = model.fit_transform(X_nonneg)
        return X_red, {
            'reconstruction_err_': model.reconstruction_err_,
            'n_components': model.n_components
        }

    elif method == 'isomap':
        model = Isomap(n_components=n_components)
        X_red = model.fit_transform(X)
        return X_red, {'n_components': n_components}

    elif method == 'tsne':
        model = TSNE(n_components=3, random_state=42, perplexity=30, init='random')
        X_red = model.fit_transform(X)
        return X_red, {'n_components': n_components}


# clustering
def apply_clustering(X_red):
    inertia_results = []
    for k in range(2, 6):
        km = KMeans(n_clusters=k, random_state=0).fit(X_red)
        inertia_results.append(km.inertia_)
    agg = AgglomerativeClustering(n_clusters=3).fit_predict(X_red)
    dbs = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_red)
    return inertia_results, agg, dbs


# classification
def train_svc(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    svc = SVC(probability=True)
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    return acc, f1


# json builders
def build_json_result(name, info, metrics):
    return {
        name: {
            **info,
            "accuracy": metrics["acc"],
            "f1_score": metrics["f1"],
        }
    }


def build_clustering_json(inertia, metrics):
    return {
        "inertia_results": inertia,
        "accuracy": metrics["acc"],
        "f1_score": metrics["f1"],
    }


# full pipeline
def full_analysis_pipeline(csv_path):
    X, y = load_prepare_data(csv_path)
    results = {"decomposition": {}, "clustering": {}}

    for method in ['pca', 'nmf', 'isomap', 'tsne']:
        try:
            if method == 'nmf':
                metrics = dict(zip(['acc', 'f1'], train_svc(X, y)))
                results["decomposition"].update(build_json_result(method, info, metrics))
            else:
                X_red, info = apply_decomposition(X, method, n_components=5)
                metrics = dict(zip(['acc', 'f1'], train_svc(X_red, y)))
                results["decomposition"].update(build_json_result(method, info, metrics))
        except Exception as e:
            print(f"{method} failed: {e}")

    try:
        X_pca, _ = apply_decomposition(X, 'pca', 5)
        inertia_results, _, _ = apply_clustering(X_pca)
        metrics = dict(zip(['acc', 'f1'], train_svc(X_pca, y)))
        results["clustering"] = build_clustering_json(inertia_results, metrics)
    except Exception as e:
        print(f"Clustering failed: {e}")

    with open("analysis_result.json", "w") as f:
        json.dump(results, f, indent=4)
    return results


# execute
result = full_analysis_pipeline("model/data/accident_data.csv")
