from django.shortcuts import render
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import time

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    df = pd.read_csv(r"C:\Users\lymon\Desktop\Lymon\ai4i2020.csv")

    for column in df.columns:
        try:
            df[column] = df[column].astype(float)
        except:
            pass

    df.drop(['UDI', 'Product ID'], axis=1, inplace=True)
    df.drop(['Type'], axis=1, inplace=True)
    df.drop(['TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1, inplace=True)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    oversamp = SMOTE(n_jobs=-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    X_train, y_train = oversamp.fit_resample(X_train, y_train)

    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    start = time.time()
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0, bootstrap=True).fit(X_train, y_train)
    end_train = time.time()
    y_predictions = model.predict(X_test)
    end_predict = time.time()

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])

    pred = model.predict([[val1, val2, val3, val4, val5]])

    result1 = "The Machine Requires Maintenance Now" if pred == [1] else "No Maintenance is required"

    # Plot 1: Feature Importance Bar Plot
    feature_importances = pd.DataFrame(model.feature_importances_,
                                       index=X.columns,
                                       columns=['Importance']).sort_values('Importance', ascending=False)

    fig1 = px.bar(feature_importances, x=feature_importances.index, y='Importance',
                  title='Feature Importance in RandomForest Model')
    fig1.update_layout(width=800, height=400)

    # Plot 2: Scatter Plot of Two Features
    fig2 = px.scatter(df, x='Air temperature [K]', y='Torque [Nm]',
                      color='Machine failure',
                      title='Air Temperature vs Torque')
    fig2.update_layout(width=800, height=400)

    # Plot 3: Histogram of a Specific Feature
    fig3 = px.histogram(df, x='Rotational speed [rpm]', nbins=50,
                        title='Distribution of Rotational Speed')
    fig3.update_layout(width=800, height=400)

    # Convert plots to HTML
    plot_div1 = fig1.to_html(full_html=False)
    plot_div2 = fig2.to_html(full_html=False)
    plot_div3 = fig3.to_html(full_html=False)

    return render(request, 'predict.html', {
        "result2": result1,
        "plot_div1": plot_div1,
        "plot_div2": plot_div2,
        "plot_div3": plot_div3
    })

