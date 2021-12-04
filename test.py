
def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''	
    return pd.read_csv(pth)
    
#test
def test_import(pth,import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        logging.info("INFO: Testing import_data asserts")

        assert os.path.exists(pth)==True, 'path does not exist'


    except FileNotFoundError as err:
        logging.error("ERROR: Testing import_data: The file wasn't found: {}".format(err))
 
    try:
        logging.info("INFO: Testing import_data read file")
        df= import_data(pth)    

        logging.info("SUCCESS: Testing import_data read file")

    except BaseException as err:
        logging.error("ERROR: Testing import_data error reading file: {}".format(err))

    try:
        assert df.shape[0] > 0,'zero records'
        assert df.shape[1] > 0,'zero records'
        logging.info("SUCCESS: Testing import_data file is not empty")

        df_test=pd.read_csv(pth)

        logging.info("SUCCESS: Testing import_data file and manual method")        
        assert_frame_equal(df,df_test)
        logging.info("SUCCESS: Testing import_data file is valid")


        return df

    except AssertionError as err:
        logging.error("ERROR: Testing import_data: The file doesn't appear to have rows and columns: {}".format(err))
    
        #return empty df
        return pd.DataFrame()   
    except BaseException as err:
        logging.error("ERROR: Testing import_data: dataframe does not exist: {}".format(err))

        #return empty df
        return pd.DataFrame()