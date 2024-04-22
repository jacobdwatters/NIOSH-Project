from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def retreive_raw_violation_data():

    FEATURES = ['VIOLATION_OCCUR_DT', 'MINE_ID', 'MINE_TYPE', 'COAL_METAL_IND', 'SIG_SUB', 'LIKELIHOOD', 
                'INJ_ILLNESS', 'NO_AFFECTED', 'NEGLIGENCE', 'VIOLATOR_VIOLATION_CNT',
                'VIOLATOR_INSPECTION_DAY_CNT']
    # CONTROLLER_ID, VIOLATOR_ID, MINE_ID, and CONTRACTOR_ID are also possibly but have many categories
    FEATURES = ['VIOLATOR_TYPE_CD', 'MINE_TYPE', 'COAL_METAL_IND',
                'VIOLATION_OCCUR_DT', 'SIG_SUB', 'PRIMARY_OR_MILL', 'VIOLATOR_VIOLATION_CNT',
                'VIOLATOR_INSPECTION_DAY_CNT']
    TARGETS = ['PROPOSED_PENALTY']

    violation_data = pd.read_csv("https://arlweb.msha.gov/OpenGovernmentData/DataSets/Violations.zip", 
                                 encoding='latin-1', compression='zip', sep='|', 
                                 usecols = [*FEATURES, *TARGETS])

    return violation_data

def retreive_raw_violation_data_contest():
    features = [
        # Temporal Features
        # "INSPECTION_BEGIN_DT", "INSPECTION_END_DT",
        # "VIOLATION_ISSUE_DT", "VIOLATION_OCCUR_DT",
        # "CAL_YR", "CAL_QTR", "FISCAL_YR", "FISCAL_QTR",
        # "ORIG_TERM_DUE_DT", "LATEST_TERM_DUE_DT",
        "VIOLATION_OCCUR_DT",
    
        # Violation Specifics
        "SIG_SUB",
        "SECTION_OF_ACT", #"PART_SECTION",
        "LIKELIHOOD", "INJ_ILLNESS",
        "NO_AFFECTED",
        "NEGLIGENCE",
    
        # Financial Features
        "PROPOSED_PENALTY", "AMOUNT_DUE",
    
        # Other Features
        "ENFORCEMENT_AREA",
        "SPECIAL_ASSESS",
        "VIOLATOR_VIOLATION_CNT", "VIOLATOR_INSPECTION_DAY_CNT",
        "LAST_ACTION_CD", "DOCKET_STATUS_CD",
        "MINE_TYPE", "COAL_METAL_IND", "PRIMARY_OR_MILL"
    ]

    targets = ['CONTESTED_IND']

    violation_data = pd.read_csv("https://arlweb.msha.gov/OpenGovernmentData/DataSets/Violations.zip", 
                                 encoding='latin-1', compression='zip', sep='|', 
                                 usecols = [*features, *targets])
    
    return violation_data

def process_raw_contest_data(violation_data):
    # Impute missing categorical values
    violation_data['MINE_TYPE'].fillna('Facility', inplace=True)
    violation_data['COAL_METAL_IND'].fillna('M', inplace=True)
    violation_data['PRIMARY_OR_MILL'].fillna('Not_Applicable', inplace=True)
    violation_data['SECTION_OF_ACT'].fillna('NoSection', inplace=True)
    violation_data['ENFORCEMENT_AREA'].fillna('Unknown', inplace=True)
    # violation_data['LAST_ACTION_CD'].fillna('Unknown', inplace=True)
    # violation_data['DOCKET_STATUS_CD'].fillna('Unknown', inplace=True)

    # Impute missing numerical values
    violation_data['VIOLATOR_VIOLATION_CNT'].fillna(0, inplace=True)
    violation_data['VIOLATOR_INSPECTION_DAY_CNT'].fillna(0, inplace=True)
    violation_data['PROPOSED_PENALTY'].fillna(0, inplace=True)
    # violation_data['AMOUNT_DUE'].fillna(0, inplace=True)
    violation_data['NO_AFFECTED'].fillna(0, inplace=True)

    # Convert datetime fields and extract temporal features
    violation_data['VIOLATION_OCCUR_DT'] = pd.to_datetime(violation_data['VIOLATION_OCCUR_DT'], format='%m/%d/%Y', exact=False)
    violation_data.reset_index(inplace=True)
    violation_data['YEAR_OCCUR'] = violation_data['VIOLATION_OCCUR_DT'].dt.year



    # Drop unnecessary columns
    violation_data = violation_data.drop(columns=['VIOLATION_OCCUR_DT'])
    violation_data = violation_data.drop(columns=['index'])

    return violation_data.dropna()



def process_raw_violation_data(violation_data):
    violation_data['MINE_TYPE'].fillna('Facility', inplace=True)
    violation_data['COAL_METAL_IND'].fillna('M', inplace=True)
    violation_data['PRIMARY_OR_MILL'].fillna('Not_Applicable', inplace=True)
    #violation_data['SIG_SUB'].fillna('N', inplace=True)
    #violation_data['LIKELIHOOD'].fillna('NoLikelihood', inplace=True)
    #violation_data['INJ_ILLNESS'].fillna('NoLostDays', inplace=True)
    #violation_data['NO_AFFECTED'].fillna(0, inplace=True)
    #violation_data['NEGLIGENCE'].fillna('NoNegligence', inplace=True)
    violation_data['VIOLATOR_VIOLATION_CNT'].fillna(0, inplace=True)
    violation_data['VIOLATOR_INSPECTION_DAY_CNT'].fillna(0, inplace=True)
    #violation_data['PROPOSED_PENALTY'].fillna(violation_data['PROPOSED_PENALTY'].mean(), inplace=True)

    violation_data['VIOLATION_OCCUR_DT'] = pd.to_datetime(violation_data['VIOLATION_OCCUR_DT'], format='%m/%d/%Y', exact=False)
    violation_data.reset_index(inplace=True)

    violation_data['YEAR_OCCUR'] = violation_data['VIOLATION_OCCUR_DT'].dt.year

    #violation_data['YEAR_OCCUR'].fillna('1999', inplace=True)
    #violation_data = violation_data.convert(numeric=True)

    violation_data = violation_data.drop(columns=['VIOLATION_OCCUR_DT'])
    violation_data = violation_data.drop(columns=['index'])
    
    return violation_data


def scale_and_encode(violation_data, to_scale=None, to_encode=None, target='PROPOSED_PENALTY', target_method=StandardScaler):
    if to_scale is None:
        to_scale = ['VIOLATOR_INSPECTION_DAY_CNT', 'VIOLATOR_VIOLATION_CNT', 'YEAR_OCCUR']
    
    if to_encode is None:
        to_encode = ['PRIMARY_OR_MILL', 'COAL_METAL_IND', 'MINE_TYPE', 'SIG_SUB', 'VIOLATOR_TYPE_CD']
    
    FEATURES = [col for col in violation_data.columns if col != target]
    TARGETS = [target]
    
    X = violation_data[FEATURES]
    y = violation_data[TARGETS]
    
    # Instantiate encoder/scaler
    scaler = StandardScaler()
    ohe = OneHotEncoder(sparse=False)

    target_scaler = target_method()
        
    

    # Scale and Encode Separate Columns
    scaled_columns  = scaler.fit_transform(X[to_scale])
    encoded_columns = ohe.fit_transform(X[to_encode])

    # Concatenate (Column-Bind) Processed Columns Back Together
    X_pre = np.concatenate([scaled_columns, encoded_columns], axis=1)
    np.nan_to_num(X_pre, copy=False)

    y_pre = target_scaler.fit_transform(y)
    
    return (X_pre, y_pre), (scaler, ohe, target_scaler)


def get_processed_violation_data(use_local=True, save_local=True, local_file="data/violations_processed.csv"):
    '''Gets dataframe of processed (but not scaled/encoded) violation data.
    Uses a local copy if possible.'''
    data = None
    if not use_local or not Path(local_file).is_file():
        data = process_raw_violation_data(retreive_raw_violation_data())
        if save_local:
            data.to_csv(local_file)
    else:
        data = pd.read_csv(local_file, index_col=0)
    return data


if __name__ == '__main__':
    data = get_processed_violation_data(use_local=True)
    #raw_data = retreive_raw_violation_data()
    #processed_data = process_raw_violation_data(raw_data)
    (X, y), (scaler, ohe, target_scaler) = scale_and_encode(data)
    print(ohe)
    print(data.shape)


def encode_and_scale(data, target, to_keep=None, categorical_cols=None, numerical_cols=None, contionous_target=None, preprocessor=None, target_transformer=None):
    """
    Encode categorical features and scale numerical features in the input data, and transform the target variable.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data to be encoded and scaled.
    target : str
        Name of the target variable in the input data.
    to_keep : list of str, optional
        List of column names to keep in the encoded and scaled data. Ignored if preprocessor is provided.
    categorical_cols : list of str, optional
        List of column names that contain categorical features. Ignored if preprocessor is provided.
    numerical_cols : list of str, optional
        List of column names that contain numerical features. Ignored if preprocessor is provided.
    contionous_target : bool, optional
        Whether the target variable is continuous or not. Ignored if target_transformer is provided.
    preprocessor : sklearn.compose.ColumnTransformer, optional
        Preprocessing pipeline to apply to the input data. If not provided, a default pipeline will be used.
    target_transformer : sklearn.base.TransformerMixin, optional
        Transformer to apply to the target variable. If not provided, a default transformer will be used.

    Returns
    -------
    X : numpy.ndarray
        Encoded and scaled feature matrix.
    y : numpy.ndarray
        Transformed target variable.
    preprocessor : sklearn.compose.ColumnTransformer
        Preprocessing pipeline used to encode and scale the input data.
    target_transformer : sklearn.base.TransformerMixin
        Transformer used to transform the target variable.

    """

    assert contionous_target is not None or target_transformer is not None, 'Either contionous_target or target_transformer must be provided.'
    assert (to_keep is not None and categorical_cols is not None and numerical_cols is not None) or preprocessor is not None, 'Either to_keep, categorical_cols, and numerical_cols must be provided, or preprocessor must be provided.'

    if preprocessor is None:

        numerical_cols_to_keep = list(set(numerical_cols).intersection(to_keep))
        categorical_cols_to_keep = list(set(categorical_cols).intersection(to_keep))

        # Define preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numerical_cols_to_keep),
                ('cat', categorical_transformer, categorical_cols_to_keep)])
        
        preprocessor.fit(data)

    # Apply transformations
    X = preprocessor.transform(data)

    # Convert target to 1D array
    y = data[target]
    y = y.values.ravel()

    # Define the transformer for target
    if target_transformer is None:
        if contionous_target:
            target_transformer = StandardScaler()
            target_transformer.fit(y[:, None])
        else:
            target_transformer = LabelEncoder()
            target_transformer.fit(y)
    
    if isinstance(target_transformer, StandardScaler):
        pass
    else:
        y = target_transformer.transform(y)

    return X, y, preprocessor, target_transformer


def encode_selected_columns(df, cols_to_encode, preprocessor=None):
    """
    OHE selected columns and returns np array.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    cols_to_encode : list of str
        List of column names to be encoded.

    Returns
    -------
    array_encoded : numpy.ndarray
        Output array with the specified columns encoded and others dropped.
    preprocessor : sklearn.compose.ColumnTransformer
        The ColumnTransformer object used for encoding.
    """
    cols_not_to_encode = [col for col in df.columns if col not in cols_to_encode]

    if preprocessor is None:
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(), cols_to_encode),
                ('drop', 'drop', cols_not_to_encode)
            ]
        )
        preprocessor.fit(df)
        
    result = preprocessor.transform(df)

    return result, preprocessor

def scale_selected_columns(df, cols_to_scale, preprocessor=None):
    """
    Scales selected numerical columns in a pandas DataFrame and returns the scaler.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    cols_to_scale : list of str
        List of column names to be scaled.

    Returns
    -------
    cols_scaled : numpy.ndarray
        Output Array with the specified columns scaled and others dropped.
    preprocessor : sklearn.compose.ColumnTransformer
        The ColumnTransformer object used for scaling.
    """
    cols_not_to_scale = [col for col in df.columns if col not in cols_to_scale]

    if preprocessor is None:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), cols_to_scale),
                ('drop', 'drop', cols_not_to_scale)
            ]
        )
        preprocessor.fit(df)
        
    cols_scaled = preprocessor.transform(df)

    return cols_scaled, preprocessor

def df_to_model_ready(df, categorical_cols, numerical_cols, target_col, preprocessor=None):
    """
    Preprocesses a pandas DataFrame to be ready for machine learning modeling.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    categorical_cols : list of str
        List of column names to be one-hot encoded.
    numerical_cols : list of str
        List of column names to be scaled.
    target_col : str
        The name of the target column.
    preprocessor : dict, optional
        A dictionary containing preprocessor objects for target, numerical and categorical columns.

    Returns
    -------
    X : numpy.ndarray
        The processed feature array.
    y : numpy.ndarray
        The processed target array.
    preprocessor : dict
        A dictionary containing preprocessor objects for target, numerical and categorical columns.
    """
    categorical_target = target_col in categorical_cols
    numerical_cols = [col for col in numerical_cols if col != target_col]
    categorical_cols = [col for col in categorical_cols if col != target_col]
    
    target = df[[target_col]].to_numpy().ravel()
    features = df.drop(target_col, axis=1)
    
    target_preprocessor = None
    numerical_preprocessor = None
    categorical_preprocessor = None
    
    if preprocessor is None:
        preprocessor = dict()
        if categorical_target:
            target_preprocessor = LabelEncoder()
            target_preprocessor.fit(target)
        else:
            target_preprocessor = StandardScaler()
            target_preprocessor.fit(target[:, np.newaxis])
        
        preprocessor['target'] = target_preprocessor

        _, numerical_preprocessor = scale_selected_columns(features, cols_to_scale=numerical_cols)
        _, categorical_preprocessor = encode_selected_columns(features, cols_to_encode=categorical_cols)

        preprocessor['numerical'] = numerical_preprocessor
        preprocessor['categorical'] = categorical_preprocessor

    processed_target = None
    if categorical_target:
        processed_target = preprocessor['target'].transform(target)
    else:
        processed_target = preprocessor['target'].transform(target[:, np.newaxis])[:, 0]
    
    scaled_features = preprocessor['numerical'].transform(features)

    encoded_cat_features = preprocessor['categorical'].transform(features)

    full_features = np.concatenate([scaled_features, encoded_cat_features], axis=1)

    return full_features, processed_target, preprocessor