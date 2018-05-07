# Utilities to help reduce clutter on main notebook

import numpy as np
import pandas as pd
import seaborn as sns

import pylab as plt
import matplotlib.patches as patches

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from plotly.offline import iplot, init_notebook_mode, plot
import plotly.graph_objs as go
import plotly.offline as offline

init_notebook_mode(connected=True)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


import GPy

GPy.plotting.change_plotting_library('plotly_offline')

sns.set(color_codes=True)

def load_data(path):
    data_raw = pd.read_csv(path).drop(columns=['Unnamed: 0'])

    # Drop some bad samples from data
    data_raw = data_raw[data_raw.number_of_rooms != ' ']
    data_raw = data_raw[data_raw.dist_center < 100]


    #print(data_raw.isnull().sum())
    # Energy class has 1105 na:s, the rest of the columns only a few. We drop the energy class column completely.

    # Drop unnecessary columns, they are still in data_raw if needed later
    data_with_neighborhood = data_raw.drop(columns=['rooms', 'energy_class'])
    data_with_neighborhood = data_with_neighborhood.dropna()
    data = data_with_neighborhood.drop(columns=['neighborhood'])
    data = data.dropna()
    
    return data, data_with_neighborhood

def plot_variables(data):
    plt.figure(figsize=(20, 25))
    plt.subplot(4,2,1)
    plt.title('Price/m^2 per year')
    plt.xlabel('Year')
    plt.ylabel('Price/m^2')
    plt.scatter(data=data, x='year', y='price_per_square_meter')

    plt.subplot(4,2,2)
    plt.xlim(0, 20)
    plt.title('Price/m^2 per distance from center')
    plt.ylabel('Price/m^2')
    plt.scatter(data=data, x='dist_center', y='price_per_square_meter')

    plt.subplot(4,2,3)
    #plt.xlim(0, 20)
    plt.title('Price/m^2 per condition')
    plt.ylabel('Price/m^2')
    sns.violinplot(data=data, x='condition', y='price_per_square_meter')

    plt.subplot(4,2,4)
    plt.xlim(60.1, 60.4)
    plt.title('Price/m^2 per latitude')
    plt.ylabel('Price/m^2')
    plt.scatter(data=data, x='lat', y='price_per_square_meter')

    plt.subplot(4,2,5)
    plt.title('Price/m^2 per floor fraction')
    plt.ylabel('Price/m^2')
    plt.scatter(data=data, x='floor_frac', y='price_per_square_meter')

    plt.subplot(4,2,6)
    plt.title('Price/m^2 per floor number')
    sns.violinplot(data=data, x='floor_num', y='price_per_square_meter', jitter=True)

    plt.subplot(4,2,7)
    plt.title('Price/m^2 per number of rooms')
    sns.violinplot(data=data, x='number_of_rooms', y='price_per_square_meter', jitter=True)

    plt.subplot(4,2,8)
    plt.title('dist_center per year')
    plt.ylim(0, 20)
    plt.scatter(data=data, x='year', y='dist_center')
    
    plt.show()
    

def plot_correlation(data):
    corr = data.corr()
    f, ax = plt.subplots(figsize=(15, 15), dpi=250)
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax)

    ax.add_patch(
        patches.Rectangle(
            (5, 0),
            1,
            14,
            fill=False,
            linewidth=2,
            color='yellow'
        )
    )
    ax.add_patch(
        patches.Rectangle(
                (0, 5),
                14,
                1,
                fill=False,
                linewidth=2,
                color='yellow'
        )
    )

    plt.title('Correlation plot')
    
def tsne_plots(data_tsne_embedded, data, k_labels, d_labels):
    plt.figure(figsize=(20, 20))
    plt.subplot(4,3,1)
    plt.title('TSNE embedding, color is log(price)')
    plt.scatter(x = data_tsne_embedded[:,0], y = data_tsne_embedded[:,1], c = np.log(data.price))
    plt.colorbar()

    plt.subplot(4,3,2)
    plt.title('TSNE embedding, color is price / m^2')
    plt.scatter(x = data_tsne_embedded[:,0], y = data_tsne_embedded[:,1], c = data.price_per_square_meter)
    plt.colorbar()

    plt.subplot(4,3,3)
    plt.title('TSNE embedding, color is number of rooms')
    plt.scatter(x = data_tsne_embedded[:,0], y = data_tsne_embedded[:,1], c = data.number_of_rooms)
    plt.colorbar()

    plt.subplot(4,3,4)
    plt.title('TSNE embedding, color is year')
    plt.scatter(x = data_tsne_embedded[:,0], y = data_tsne_embedded[:,1], c = data.year)
    plt.colorbar()

    plt.subplot(4,3,5)
    plt.title('TSNE embedding, color is condition')
    plt.scatter(x = data_tsne_embedded[:,0], y = data_tsne_embedded[:,1], c = data.condition)
    plt.colorbar()

    plt.subplot(4,3,6)
    plt.title('TSNE embedding, color is floor number')
    plt.scatter(x = data_tsne_embedded[:,0], y = data_tsne_embedded[:,1], c = data.floor_num)
    plt.colorbar()

    plt.subplot(4,3,7)
    plt.title('TSNE embedding, color is log distance from center')
    plt.scatter(x = data_tsne_embedded[:,0], y = data_tsne_embedded[:,1], c = np.log(data.dist_center))
    plt.colorbar()


    plt.subplot(4,3,8)
    plt.title('TSNE embedding, color is kmeans cluster label')
    plt.scatter(x = data_tsne_embedded[:,0], y = data_tsne_embedded[:,1], c = k_labels)
    plt.colorbar()

    plt.subplot(4,3,9)
    plt.title('TSNE embedding, color is DBSCAN cluster label')
    plt.scatter(x = data_tsne_embedded[:,0], y = data_tsne_embedded[:,1], c = d_labels)
    plt.colorbar()

    
def pca_plots(data_pca_model, data_pca_embedded, data, labels):
    components = data_pca_model.components_
    ratios = data_pca_model.explained_variance_ratio_
    
    print('Explained variance for the first two principal components:')
    print(ratios[0]+ ratios[1])

    plt.figure(figsize=(20,15))
    plt.subplot(2,2,1)
    plt.bar(np.arange(0,len(ratios)), ratios)
    plt.title('Explained variance')

    plt.subplot(2,2,2)
    plt.title('First two principal components, color = price/m^2')
    plt.scatter(x = data_pca_embedded[:,0], y = data_pca_embedded[:,1], c = data.price_per_square_meter)
    plt.colorbar()

    plt.subplot(2,2,3)
    plt.title('First principal component vs price per square meter, color = clustering labels')
    plt.scatter(x = data_pca_embedded[:,0], y = data.price_per_square_meter, c = labels)

    plt.subplot(2,2,4)
    plt.title('Second principal component vs price per square meter, color = clustering labels')
    plt.scatter(x = data_pca_embedded[:,1], y = data.price_per_square_meter, c = labels)
    plt.colorbar()
    
    
def model_information(model, x_test=None, y_test=None, plot_test_data = False, plot_data=True):
    print(model)
 
    log_marginal_likelihood = model.log_likelihood()
    print('\nLog marginal likelihood:')
    print(log_marginal_likelihood)
     
    if x_test is not None and y_test is not None:
        predictions = model.predict(x_test)[0]
        print (predictions.shape)
        mean_test_error = np.mean((predictions - y_test)**2)
        print('\nMean squared test error:')
        print(mean_test_error)
        
        print("\nMean absolute test error: %.2f"
              % mean_absolute_error(y_test, predictions))

        mlppd = np.mean(model.log_predictive_density(x_test, y_test))
        print('\nMLPPD:')
        print(mlppd)
        
    if plot_data:
        fig = model.plot()

        if plot_test_data:
            fig_data = fig[0]['data']
            test_info = {'type': 'scatter', 
                         'x': np.squeeze(x_test,1), 
                         'y': np.squeeze(y_test,1), 
                         'mode': 'markers', 
                         'showlegend': True, 
                         'marker': {'color': 'red', 'colorscale': None}, 
                         'name': 'Test'}
            fig_data.append(test_info)
            fig[0]['data'] = fig_data


        GPy.plotting.show(fig)

def plotLinearData(x_train, y_train, x_test, y_test, orders):
    for order in range(1,orders+1):

        print("Linear regression fit of order {}".format(order))
        poly = PolynomialFeatures(order)
        x_train_poly = poly.fit_transform(x_train)
        x_test_poly = poly.fit_transform(x_test)

        ln = LinearRegression()

        ln.fit(x_train_poly, y_train)

        y_pred = ln.predict(x_test_poly)

        # For plotting
        x_plot = np.arange(np.min(x_train),np.max(x_train), 1)
        y_plot = ln.predict(poly.fit_transform(np.expand_dims(x_plot, 1)))


        # The coefficients
        print('Coefficients: \n', ln.coef_)
        print("Mean squared error: %.2f"
              % mean_squared_error(y_test, y_pred))
        print("Mean absolute error: %.2f"
              % mean_absolute_error(y_test, y_pred))
        print('Variance score: %.2f' % r2_score(y_test, y_pred))

        # Plot outputs
        p1 = go.Scatter(x=x_test.values.flatten(), 
                        y=y_test.values.flatten(), 
                        mode='markers',
                        marker=dict(color='red')
                       )
        
        p2 = go.Scatter(x=x_train.values.flatten(), 
                        y=y_train.values.flatten(), 
                        mode='markers',
                        marker=dict(color='black')
                       )

        p3 = go.Scatter(x=x_plot, 
                        y=y_plot,
                        mode='lines',
                        line=dict(color='blue', width=3)
                        )

        layout = go.Layout(xaxis=dict(ticks='', showticklabels=True,
                                      zeroline=True),
                           yaxis=dict(ticks='', showticklabels=True,
                                      zeroline=True),
                           showlegend=False, hovermode='closest')

        fig = go.Figure(data=[p1, p2, p3], layout=layout)


        iplot(fig)

    
    