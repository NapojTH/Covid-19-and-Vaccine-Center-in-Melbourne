# Covid-19-and-Vaccine-Center-in-Melbourne

The following tasks include an integration and reshaping of the Covid-19 cases and trips toward the vaccine central in Melbourne data from various sources (PDF of LGA list, xml and json file of individual location, shp.file containing suburb, txt file containing trip information and Covid-19 cases from https://covidlive.com.au/).
(More detail available in Jupyter Notebook file)

## Summary

### Integration of information
Several methods were used for preprocess the data from different sources. Some of the data did not need to be edited before store into a dataframe, while some information need to be calculated. For example, the distance_to_closet_train_station column need to be calculated based on the lattitude and longitude between the individuals habitat and the closet_train_station_id. 

All of the column except 30_sep_cases, last_14_days_cases, last_30_days_cases, and last_60_days_cases are extracted from varrious sources. Then, all information is integrated into a single dataframe by using Beautifulsoup, PyPDF, Json, geopandas and txt file parsing. While, the Covid-19 cases from Victoria is extracted from web scrapping packages, BeautifulSoup

Once all the information is stored within the dataframe, the csv file containing all the information will be created. This csv file will be used for making a linear regression model. 

### Reshaping
In the following task, the csv file created in the previous task will be used for creating regression model. When the file was read, the columns which will be used for this task are 30_sep_cases, last_14_days_cases, last_30_days_cases, and last_60_days_cases. Before the model is created, the data will be reshaped by using the following approachs: Square transformation, Root transformation, Log transformation and Normalisation. Once each approachs were applied on the data separately, the transformed/normalised data will be used for fitting in the models. Then, the comparison of the model performance will be performed by identifying their r-square and root mean square error(RMSE).

Here is the result of the models with serveral transformation approachs:
1. Data without any transformation: R-square = 0.965, RMSE = 8.868
2. Data with Root Transformation: R-square = 0.871, RMSE = 17.176
3. Data with Square Transformation: R-square = 0.805 RMSE = 21.108
4. Data with Log Transformation: No result because the data got a value of zero in across the multiple columns
5. Normalisation of data based on z-score : R-square = 0.965, RMSE = 0.173

