TITLE = "**Insurace Prediction**"


OBJECTIVE = """
### Objective
The following application tries to showcase the way in which I, Andres Felipe Tellez would solve and deploy a simple ML Web Application using the *streamlit* library. Specifically, this app predicts insurance costs based on age, sex, bmi index, number of children, whether the pacient is a smoker and region from wich this one lives at.
"""


IMAGE = "https://media.istockphoto.com/id/1199060494/photo/insurance-protecting-family-health-live-house-and-car-concept.jpg?s=612x612&w=0&k=20&c=W8bPvwF5rk7Rm2yDYnMyFhGXZfNqK4bUPlDcRpKVsB8="


NOTE = """
**Note:** CRISP-DM methodology was implemented.

### Let's take a look at the data
To begin with our application, we will look at how the first 5 observations look like:

"""


MISSING_VALUES = "It is clear that there are no missing values in the dataset."


SUMMARY_STATISTICS_TITLE = "#### Summary Statistics"


GRAPH = """
#### Graphs

##### Continous variables distributions

The first thing I want to be able to capture are the distributions of all the variables with the model that I will be creating. I also want to make sure that there is no need for that balancing strategies.
"""


RELATIONSHIP_WITH_CHARGES = "##### Relationship with charges"


DIS_VAR_DISTR = "##### Discrete variables distributions"


BOX = "##### Box and whiskers plot"


CORR = """
As all variables are now numerical, we can proceed to look at the correlation between variables.

#### Correlation         
"""


DATA_TRANS = """#### Data transformation

If we take a look at the insurance dataframe, we see that 4 out the 6 dependent variables are categorical. We will then perform some featuring engineering using hot encoder so that all this becomes columns. Denote that to avoid any problem of multicolinearity, we will remove one attribute for each variable that has more than two possible values.          

The data will now look as follows:
"""


MODEL = """
        #### Building the model
        ##### Data sets

        We will begin the construction of the model by creating two different sets of data leveraging the library of *scikit-learn*.
        
        We have also stated that 20% of the data will be used for testing the model and 80% for training.

        ##### LASSO

        In statistics and machine learning, lasso (least absolute shrinkage and selection operator) is a regression analysis method that performs both variable selection and regularization in order to enhance the prediction accuracy and interpretability of the resulting statistical model.

"""