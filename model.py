import time,os,re,csv,uuid,joblib,pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, SGDRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error as mse
from logger import update_predict_log, update_train_log
from data_ingestion import fetch_data, fetch_ts, engineer_features, convert_to_ts

with open('__version__','r+') as f:
    MODEL_VERSION = f.read()
    f.close

MODEL_DIR = "models"
MODEL_VERSION_NOTE = "supervised learing model for time-series"

def model_compare(data_dir,country='United Kingdom'):
    '''
    train all models for one country using gridsearch, return a
    df that compares the performance of models
    '''

    print('Ingesting data')
    df = fetch_data(data_dir)
    df_country = convert_to_ts(df,country)
    X,y,dates = engineer_features(df_country)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        shuffle=True, random_state=42)
    # define all pipelines and the param grids
    pipe_lr = Pipeline(steps=[('scaler', StandardScaler()),
                            ('lr', ElasticNet())])
    pipe_sgd = Pipeline(steps=[('scaler', StandardScaler()),
                            ('sgd', SGDRegressor())])
    pipe_svr = Pipeline(steps=[('scaler', StandardScaler()),
                            ('svr', SVR())])
    pipe_rf = Pipeline(steps=[('scaler', StandardScaler()),
                            ('rf', RandomForestRegressor())])
    pipe_gbt = Pipeline(steps=[('scaler', StandardScaler()),
                            ('gbt', GradientBoostingRegressor())])
    param_grid_lr = {
        'lr__max_iter':[10000],
        'lr__alpha': np.logspace(-3,0,5),
        'lr__l1_ratio': np.linspace(0,1,5)
        }
    param_grid_sgd = {
        'sgd__penalty': ['elasticnet'],
        'sgd__alpha':  np.logspace(-4,1,5),
        'sgd__l1_ratio':np.linspace(0,1,5),
        'sgd__max_iter':np.linspace(50,250,5,dtype='int'),
        'sgd__learning_rate':['optimal','invscaling']
        }
    param_grid_svr = {
        'svr__kernel': ['linear', 'poly', 'rbf','sigmoid'],
        'svr__C': np.logspace(-2,2,5),
        'svr__gamma':np.logspace(-3,0,4),
        }
    param_grid_rf = {
        'rf__n_estimators': np.linspace(25,100,4,dtype='int'),
        'rf__max_depth':np.linspace(6,15,4,dtype='int'),
        'rf__min_samples_split':np.linspace(2,8,4,dtype='int')
        }
    param_grid_gbt = {
        'gbt__learning_rate': np.logspace(-3,-1.5,5),
        'gbt__n_estimators':  np.linspace(25,100,4,dtype='int'),
        'gbt__max_depth':np.linspace(6,15,4,dtype='int'),
        'gbt__min_samples_split':np.linspace(2,8,4,dtype='int'),
        }
    all_pipes={
        pipe_lr:param_grid_lr,
        pipe_sgd:param_grid_sgd,
        pipe_svr:param_grid_svr,
        pipe_rf:param_grid_rf,
        pipe_gbt:param_grid_gbt
    }

    # train each model
    time_start_all = time.time()
    results = []
    for pipe in all_pipes:
        time_start = time.time()

        pipe_name = '->'.join([step[0] for step in pipe.steps])
        print(f'Training {pipe_name}')
        grid = GridSearchCV(pipe,param_grid=all_pipes[pipe],cv=5,n_jobs=-1,verbose=0)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        run_time = time.time()-time_start

        rmse = mse(y_test,y_pred,squared=False)
        comb = 1
        for param in [*all_pipes[pipe].values()]:
            comb = comb * len(param)
        # divide the total run time by the numbe of possible combinations of params
        avg_time = run_time / comb
        results.append([pipe_name,rmse,run_time,avg_time,grid.best_params_])

    run_time_all = time.time()-time_start_all
    print(f'Training finished! Total training time {round(run_time_all)}s')

    df = pd.DataFrame(results,columns=['pipeline','test_rmse','total_time','avg_time','best_params'])

    return df

def _model_train(df,tag,test=False):
    """
    example funtion to train model

    The 'test' flag when set to 'True':
        (1) subsets the data and serializes a test version
        (2) specifies that the use of the 'test' log file

    """

    ## start timer for runtime
    time_start = time.time()

    X,y,dates = engineer_features(df)

    if test:
        n_samples = int(np.round(0.3 * X.shape[0]))
        subset_indices = np.random.choice(np.arange(X.shape[0]),n_samples,
                                          replace=False).astype(int)
        mask = np.in1d(np.arange(y.size),subset_indices)
        y=y[mask]
        X=X[mask]
        dates=dates[mask]

    ## Perform a train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        shuffle=True, random_state=42)
    ## train a random forest model
    param_grid_rf = {
        'rf__n_estimators': np.linspace(25,100,4,dtype='int'),
        'rf__max_depth':np.linspace(6,15,4,dtype='int'),
        'rf__min_samples_split':np.linspace(2,8,4,dtype='int')
        }

    pipe_rf = Pipeline(steps=[('scaler', StandardScaler()),
                              ('rf', RandomForestRegressor())])

    grid = GridSearchCV(pipe_rf, param_grid=param_grid_rf, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    eval_rmse =  round(np.sqrt(mse(y_test,y_pred)))

    ## retrain using all data
    grid.fit(X, y)
    model_name = re.sub("\.","_",str(MODEL_VERSION))
    if test:
        saved_model = os.path.join(MODEL_DIR,
                                   f"test-{tag}-{model_name}.joblib")
        print("... saving test version of model: {}".format(saved_model))
    else:
        saved_model = os.path.join(MODEL_DIR,
                                   f"sl-{tag}-{model_name}.joblib")
        print("... saving model: {}".format(saved_model))

    joblib.dump(grid,saved_model)

    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update log
    update_train_log(tag,(str(dates[0]),str(dates[-1])),{'rmse':eval_rmse},runtime,
                     MODEL_VERSION, MODEL_VERSION_NOTE,test=test)

def model_train(data_dir,test=False):
    """
    funtion to train model given a df

    'mode' -  can be used to subset data essentially simulating a train
    """

    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    if test:
        print("... test flag on")
        print("... subseting data")
        print("... subseting countries")

    ## fetch time-series formatted data
    ts_data = fetch_ts(data_dir)

    print("... start training")
    ## train a different model for each data sets
    for country,df in ts_data.items():

        if test and country not in ['united_kingdom']:
            continue

        _model_train(df,country,test=test)

def model_load(prefix='sl',data_dir=None,training=True,save_pickle=False):
    """
    example funtion to load model

    The prefix allows the loading of different models
    """

    if not data_dir:
        data_dir = os.path.join(".","data","cs-train")

    models = [f for f in os.listdir(os.path.join(".","models")) if re.search("sl",f)]

    if len(models) == 0:
        raise Exception(f"Models with prefix '{prefix}' cannot be found did you train?")

    all_models = {}
    for model in models:
        all_models[re.split("-",model)[1]] = joblib.load(os.path.join(".","models",model))

    ## load data
    ts_data = fetch_ts(data_dir)
    all_data = {}
    for country, df in ts_data.items():
        X,y,dates = engineer_features(df,training=training)
        dates = np.array([str(d) for d in dates])
        all_data[country] = {"X":X,"y":y,"dates": dates}

    if save_pickle:
        version_ = re.sub("\.","_",str(MODEL_VERSION))
        pickle.dump((all_data, all_models), open(os.path.join("models",f"all_data_model-{version_}.pickle"), "wb" ))
        print('Pickle file saved.')
    return(all_data, all_models)

def model_predict(country,year,month,day,all_models=None,test=False,from_pickle=False):
    """
    example funtion to predict from model
    """

    ## start timer for runtime
    time_start = time.time()

    # Load all data & models from a pickle file can speed things up a lot, great for the web app
    if from_pickle:
        version_ = re.sub("\.","_",str(MODEL_VERSION))
        all_data, all_models = pickle.load(open(os.path.join("models",f"all_data_model-{version_}.pickle"), "rb" ) )
    else:
        if not all_models:
            all_data,all_models = model_load(training=False)

    ## input checks
    if country not in all_models.keys():
        raise Exception(f"ERROR (model_predict) - model for country '{country}' could not be found")

    for d in [year,month,day]:
        if re.search("\D",d):
            raise Exception("ERROR (model_predict) - invalid year, month or day")

    ## load data
    model = all_models[country]
    data = all_data[country]

    ## check date
    target_date = f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}"
    print(target_date)

    if target_date not in data['dates']:
        raise Exception(f"ERROR (model_predict) - date {target_date} not in range {data['dates'][0]}-{data['dates'][-1]}")
    date_indx = np.where(data['dates'] == target_date)[0][0]
    query = data['X'].iloc[[date_indx]]

    ## sainty check
    if data['dates'].shape[0] != data['X'].shape[0]:
        raise Exception("ERROR (model_predict) - dimensions mismatch")

    ## make prediction and gather data for log entry
    y_pred = model.predict(query)
    y_proba = None
    if 'predict_proba' in dir(model) and 'probability' in dir(model):
        if model.probability == True:
            y_proba = model.predict_proba(query)


    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update predict log
    update_predict_log(country,y_pred,y_proba,target_date,
                       runtime, MODEL_VERSION, test=test)

    return({'y_pred':y_pred,'y_proba':y_proba})

def update_version():
    with open('__version__','r+') as f:
        MODEL_VERSION = f.read()
        f.seek(0)
        major_version, minor_version = MODEL_VERSION.split('.')
        new_version = '.'.join([major_version,str(int(minor_version)+1)])
        f.write(new_version)
        f.close
