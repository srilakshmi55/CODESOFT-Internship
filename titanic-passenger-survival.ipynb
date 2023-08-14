{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "35f94433",
   "metadata": {
    "papermill": {
     "duration": 0.018443,
     "end_time": "2023-08-08T14:50:43.319253",
     "exception": false,
     "start_time": "2023-08-08T14:50:43.300810",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# The Prediction of Titanic Passengers Survival"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ed26cdd5",
   "metadata": {
    "papermill": {
     "duration": 0.018207,
     "end_time": "2023-08-08T14:50:43.356127",
     "exception": false,
     "start_time": "2023-08-08T14:50:43.337920",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "**Dataset and Problem Statement**\n",
    "\n",
    "The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.\n",
    "\n",
    "One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.\n",
    "\n",
    "Titanic dataset contains information about the people involved in the Titanic shipwreck.\n",
    "\n",
    "**Predict if a passenger survived the sinking of the Titanic or not.**\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5982e88c",
   "metadata": {
    "papermill": {
     "duration": 0.017643,
     "end_time": "2023-08-08T14:50:43.394042",
     "exception": false,
     "start_time": "2023-08-08T14:50:43.376399",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "**Variables Description**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5bb866d8",
   "metadata": {
    "papermill": {
     "duration": 0.017586,
     "end_time": "2023-08-08T14:50:43.430662",
     "exception": false,
     "start_time": "2023-08-08T14:50:43.413076",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "PassengerID : ID of the Passenger.\n",
    "\n",
    "Survived: Survival (0 = No; 1 = Yes)\n",
    "\n",
    "Pclass: Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)\n",
    "\n",
    "Name : Name of the Passenger\n",
    "\n",
    "Sex: Sex of the Passenger (Female / Male)\n",
    "\n",
    "Age: Age of the Passenger.\n",
    "\n",
    "Sibsp: Number of siblings/spouses aboard\n",
    "\n",
    "Parch: Number of parents/children aboard\n",
    "\n",
    "Ticket : Ticket number.\n",
    "\n",
    "Fare: Passenger fare (British pound)\n",
    "\n",
    "Cabin: Cabin number\n",
    "\n",
    "Embarked: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "d1778417",
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:43.468528Z",
     "iopub.status.busy": "2023-08-08T14:50:43.467970Z",
     "iopub.status.idle": "2023-08-08T14:50:43.490325Z",
     "shell.execute_reply": "2023-08-08T14:50:43.489405Z"
    },
    "papermill": {
     "duration": 0.044533,
     "end_time": "2023-08-08T14:50:43.493130",
     "exception": false,
     "start_time": "2023-08-08T14:50:43.448597",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/kaggle/input/test-file/tested.csv\n"
     ]
    }
   ],
   "source": [
    "# This Python 3 environment comes with many helpful analytics libraries installed\n",
    "# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python\n",
    "# For example, here's several helpful packages to load\n",
    "\n",
    "import numpy as np # linear algebra\n",
    "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n",
    "\n",
    "# Input data files are available in the read-only \"../input/\" directory\n",
    "# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n",
    "\n",
    "import os\n",
    "for dirname, _, filenames in os.walk('/kaggle/input'):\n",
    "    for filename in filenames:\n",
    "        print(os.path.join(dirname, filename))\n",
    "\n",
    "# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using \"Save & Run All\" \n",
    "# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "48d5e4a6",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:43.532539Z",
     "iopub.status.busy": "2023-08-08T14:50:43.531588Z",
     "iopub.status.idle": "2023-08-08T14:50:43.560644Z",
     "shell.execute_reply": "2023-08-08T14:50:43.559530Z"
    },
    "papermill": {
     "duration": 0.051343,
     "end_time": "2023-08-08T14:50:43.563274",
     "exception": false,
     "start_time": "2023-08-08T14:50:43.511931",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "df = pd.read_csv(\"/kaggle/input/test-file/tested.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "ac33ef83",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:43.602417Z",
     "iopub.status.busy": "2023-08-08T14:50:43.601662Z",
     "iopub.status.idle": "2023-08-08T14:50:43.628701Z",
     "shell.execute_reply": "2023-08-08T14:50:43.627631Z"
    },
    "papermill": {
     "duration": 0.049579,
     "end_time": "2023-08-08T14:50:43.631370",
     "exception": false,
     "start_time": "2023-08-08T14:50:43.581791",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>892</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Kelly, Mr. James</td>\n",
       "      <td>male</td>\n",
       "      <td>34.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>330911</td>\n",
       "      <td>7.8292</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>893</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Wilkes, Mrs. James (Ellen Needs)</td>\n",
       "      <td>female</td>\n",
       "      <td>47.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>363272</td>\n",
       "      <td>7.0000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>894</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>Myles, Mr. Thomas Francis</td>\n",
       "      <td>male</td>\n",
       "      <td>62.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>240276</td>\n",
       "      <td>9.6875</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass                              Name     Sex  \\\n",
       "0          892         0       3                  Kelly, Mr. James    male   \n",
       "1          893         1       3  Wilkes, Mrs. James (Ellen Needs)  female   \n",
       "2          894         0       2         Myles, Mr. Thomas Francis    male   \n",
       "\n",
       "    Age  SibSp  Parch  Ticket    Fare Cabin Embarked  \n",
       "0  34.5      0      0  330911  7.8292   NaN        Q  \n",
       "1  47.0      1      0  363272  7.0000   NaN        S  \n",
       "2  62.0      0      0  240276  9.6875   NaN        Q  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "0c118561",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:43.671038Z",
     "iopub.status.busy": "2023-08-08T14:50:43.670285Z",
     "iopub.status.idle": "2023-08-08T14:50:43.677520Z",
     "shell.execute_reply": "2023-08-08T14:50:43.676317Z"
    },
    "papermill": {
     "duration": 0.029959,
     "end_time": "2023-08-08T14:50:43.679850",
     "exception": false,
     "start_time": "2023-08-08T14:50:43.649891",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(418, 12)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "562169d7",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:43.719803Z",
     "iopub.status.busy": "2023-08-08T14:50:43.719124Z",
     "iopub.status.idle": "2023-08-08T14:50:43.749679Z",
     "shell.execute_reply": "2023-08-08T14:50:43.748041Z"
    },
    "papermill": {
     "duration": 0.053401,
     "end_time": "2023-08-08T14:50:43.752316",
     "exception": false,
     "start_time": "2023-08-08T14:50:43.698915",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 418 entries, 0 to 417\n",
      "Data columns (total 12 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  418 non-null    int64  \n",
      " 1   Survived     418 non-null    int64  \n",
      " 2   Pclass       418 non-null    int64  \n",
      " 3   Name         418 non-null    object \n",
      " 4   Sex          418 non-null    object \n",
      " 5   Age          332 non-null    float64\n",
      " 6   SibSp        418 non-null    int64  \n",
      " 7   Parch        418 non-null    int64  \n",
      " 8   Ticket       418 non-null    object \n",
      " 9   Fare         417 non-null    float64\n",
      " 10  Cabin        91 non-null     object \n",
      " 11  Embarked     418 non-null    object \n",
      "dtypes: float64(2), int64(5), object(5)\n",
      "memory usage: 39.3+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "4f79ecda",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:43.792320Z",
     "iopub.status.busy": "2023-08-08T14:50:43.791305Z",
     "iopub.status.idle": "2023-08-08T14:50:43.800470Z",
     "shell.execute_reply": "2023-08-08T14:50:43.799635Z"
    },
    "papermill": {
     "duration": 0.031475,
     "end_time": "2023-08-08T14:50:43.802679",
     "exception": false,
     "start_time": "2023-08-08T14:50:43.771204",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId      0\n",
       "Survived         0\n",
       "Pclass           0\n",
       "Name             0\n",
       "Sex              0\n",
       "Age             86\n",
       "SibSp            0\n",
       "Parch            0\n",
       "Ticket           0\n",
       "Fare             1\n",
       "Cabin          327\n",
       "Embarked         0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "8332595e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:43.843237Z",
     "iopub.status.busy": "2023-08-08T14:50:43.842147Z",
     "iopub.status.idle": "2023-08-08T14:50:43.852110Z",
     "shell.execute_reply": "2023-08-08T14:50:43.851200Z"
    },
    "papermill": {
     "duration": 0.032577,
     "end_time": "2023-08-08T14:50:43.854415",
     "exception": false,
     "start_time": "2023-08-08T14:50:43.821838",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# imputing age with median\n",
    "missing_col = ['Age','Fare']\n",
    " \n",
    "for i in missing_col:\n",
    "    df.loc[df.loc[:,i].isnull(),i]=df.loc[:,i].median()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "c4c26fd7",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:43.894337Z",
     "iopub.status.busy": "2023-08-08T14:50:43.893939Z",
     "iopub.status.idle": "2023-08-08T14:50:43.931185Z",
     "shell.execute_reply": "2023-08-08T14:50:43.929878Z"
    },
    "papermill": {
     "duration": 0.059855,
     "end_time": "2023-08-08T14:50:43.933737",
     "exception": false,
     "start_time": "2023-08-08T14:50:43.873882",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>418.000000</td>\n",
       "      <td>418.000000</td>\n",
       "      <td>418.000000</td>\n",
       "      <td>418.000000</td>\n",
       "      <td>418.000000</td>\n",
       "      <td>418.000000</td>\n",
       "      <td>418.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>1100.500000</td>\n",
       "      <td>0.363636</td>\n",
       "      <td>2.265550</td>\n",
       "      <td>29.599282</td>\n",
       "      <td>0.447368</td>\n",
       "      <td>0.392344</td>\n",
       "      <td>35.576535</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>120.810458</td>\n",
       "      <td>0.481622</td>\n",
       "      <td>0.841838</td>\n",
       "      <td>12.703770</td>\n",
       "      <td>0.896760</td>\n",
       "      <td>0.981429</td>\n",
       "      <td>55.850103</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>892.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.170000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>996.250000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>23.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7.895800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>1100.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>27.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14.454200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>1204.750000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>35.750000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>31.471875</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1309.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>76.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>9.000000</td>\n",
       "      <td>512.329200</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       PassengerId    Survived      Pclass         Age       SibSp  \\\n",
       "count   418.000000  418.000000  418.000000  418.000000  418.000000   \n",
       "mean   1100.500000    0.363636    2.265550   29.599282    0.447368   \n",
       "std     120.810458    0.481622    0.841838   12.703770    0.896760   \n",
       "min     892.000000    0.000000    1.000000    0.170000    0.000000   \n",
       "25%     996.250000    0.000000    1.000000   23.000000    0.000000   \n",
       "50%    1100.500000    0.000000    3.000000   27.000000    0.000000   \n",
       "75%    1204.750000    1.000000    3.000000   35.750000    1.000000   \n",
       "max    1309.000000    1.000000    3.000000   76.000000    8.000000   \n",
       "\n",
       "            Parch        Fare  \n",
       "count  418.000000  418.000000  \n",
       "mean     0.392344   35.576535  \n",
       "std      0.981429   55.850103  \n",
       "min      0.000000    0.000000  \n",
       "25%      0.000000    7.895800  \n",
       "50%      0.000000   14.454200  \n",
       "75%      0.000000   31.471875  \n",
       "max      9.000000  512.329200  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "bcf86d3c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:43.973967Z",
     "iopub.status.busy": "2023-08-08T14:50:43.973572Z",
     "iopub.status.idle": "2023-08-08T14:50:43.986488Z",
     "shell.execute_reply": "2023-08-08T14:50:43.985367Z"
    },
    "papermill": {
     "duration": 0.035673,
     "end_time": "2023-08-08T14:50:43.988906",
     "exception": false,
     "start_time": "2023-08-08T14:50:43.953233",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId    418\n",
       "Survived         2\n",
       "Pclass           3\n",
       "Name           418\n",
       "Sex              2\n",
       "Age             79\n",
       "SibSp            7\n",
       "Parch            8\n",
       "Ticket         363\n",
       "Fare           169\n",
       "Cabin           76\n",
       "Embarked         3\n",
       "dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.nunique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "b89bfa59",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:44.030025Z",
     "iopub.status.busy": "2023-08-08T14:50:44.029384Z",
     "iopub.status.idle": "2023-08-08T14:50:44.037725Z",
     "shell.execute_reply": "2023-08-08T14:50:44.036523Z"
    },
    "papermill": {
     "duration": 0.03143,
     "end_time": "2023-08-08T14:50:44.039985",
     "exception": false,
     "start_time": "2023-08-08T14:50:44.008555",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId      int64\n",
       "Survived         int64\n",
       "Pclass           int64\n",
       "Name            object\n",
       "Sex             object\n",
       "Age            float64\n",
       "SibSp            int64\n",
       "Parch            int64\n",
       "Ticket          object\n",
       "Fare           float64\n",
       "Cabin           object\n",
       "Embarked        object\n",
       "dtype: object"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "312fa73b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:44.080915Z",
     "iopub.status.busy": "2023-08-08T14:50:44.080270Z",
     "iopub.status.idle": "2023-08-08T14:50:46.059037Z",
     "shell.execute_reply": "2023-08-08T14:50:46.058011Z"
    },
    "papermill": {
     "duration": 2.002286,
     "end_time": "2023-08-08T14:50:46.061698",
     "exception": false,
     "start_time": "2023-08-08T14:50:44.059412",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/conda/lib/python3.10/site-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.23.5\n",
      "  warnings.warn(f\"A NumPy version >={np_minversion} and <{np_maxversion}\"\n"
     ]
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt #(For Visualizations)\n",
    "import seaborn as sns #(For Visualizations)\n",
    "from sklearn.impute import SimpleImputer #(For imputation)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "7ffba33c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:46.103089Z",
     "iopub.status.busy": "2023-08-08T14:50:46.102367Z",
     "iopub.status.idle": "2023-08-08T14:50:46.364155Z",
     "shell.execute_reply": "2023-08-08T14:50:46.362904Z"
    },
    "papermill": {
     "duration": 0.285379,
     "end_time": "2023-08-08T14:50:46.366691",
     "exception": false,
     "start_time": "2023-08-08T14:50:46.081312",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Survived', ylabel='count'>"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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ePXu3Vsvv/wy9wQBQYjYAdCmxo8fr1OnTmnr1q0aMWKEdu3apVtuuUX5+fk/6zx9+/b12540aZL+/e9/a9++fZKk1157TTfffLNuvPHGRp9/5513KiQkRFu3bpUkvfHGG4qIiFBGRoYkqbi4WEePHlVERITvilRUVJS+++47ff3116qqqlJpaakGDBjgO2dISEiDuQAEHrEDoM117txZw4cP1xNPPKGioiJNmzZNTz75pDp0+P6fpP+976Wurq7Rc3Tt2tVvOz4+Xunp6b6rO6+//volr+pI39+wfPfdd/vWb9iwQRMnTvTd6Hzx4kWlpqbqwIEDfo+vvvpKkydPbvqbB9DmiB0AAXfjjTfq3Llzio2NlSSVlpb6jv3vzco/ZcqUKdq0aZM++ugjff3115o0adJPrt++fbsOHz6snTt3asqUKb5jt9xyi44cOaK4uDhdf/31fg+HwyGHw6H4+HjflSRJunDhgoqLi694XgBtg9gB0GZOnz6tIUOGaP369Tp48KBKSkr0t7/9TYsXL9aYMWMUFham/v3767nnntNnn32m3bt36//+7/+u+Pzjxo2Tx+PRo48+qvT0dF1zzTWXXT9o0CA5nU5NmTJF1157rfr37+87NmXKFMXExGjMmDHas2ePSkpKVFhYqMcee0wnT56UJD322GN67rnntHnzZn3xxReaMWNGq//QQgA/H7EDoM2Eh4erX79+Wrp0qQYOHKjk5GQtXLhQDz30kJYvXy5JWrVqlerq6tS3b1899thjeuaZZ674/JGRkRo1apT++c9/+l2luRSbzaZ777230fVdunTR7t271bNnT40bN069e/fWAw88oJqaGkVGRkqS5syZo/vvv1/Tpk3TgAEDFBERobvuuutn/I0AaAs2ix8KAQAADMaVHQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEb7f9Y9qRDNmzHzAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x='Survived',data=df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "e5ef0595",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:46.411998Z",
     "iopub.status.busy": "2023-08-08T14:50:46.411527Z",
     "iopub.status.idle": "2023-08-08T14:50:46.665628Z",
     "shell.execute_reply": "2023-08-08T14:50:46.664433Z"
    },
    "papermill": {
     "duration": 0.279609,
     "end_time": "2023-08-08T14:50:46.668276",
     "exception": false,
     "start_time": "2023-08-08T14:50:46.388667",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Sex', ylabel='Survived'>"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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dtlll3zqU59KS0tL+diCBQvy+uuvZ8yYMeV9DQ0NGTp0aObNm7fO12xvb09bW1uHDQAAYH11KqRGjx6dyy67rPy4qqoqy5cvz5e+9KV85CMf6arZMm7cuNxwww2555578rWvfS3z58/PAQcckPb29iRJc3NzevXqle22267D8+rq6tLc3LzO1505c2Zqa2vLW2NjY5fNDAAAbPo69ad9l156afbff/984AMfyGuvvZbjjz8+Tz/9dAYMGJCbbrqpy4Y79thjyz8PHTo0e+65ZwYPHpzbbrstRx999DqfVyqVyncSXJvp06dn6tSp5cdtbW1iCgAAWG+dCqmGhoYsXLgwN910U379619n9erVmTBhQj7xiU+kd+/eXT1j2aBBgzJ48OA8/fTTSZL6+vqsXLkyS5cu7XBVqqWlJSNGjFjn61RXV6e6unqDzQkAAGzaOhVSr776arbeeuuceuqpOfXUU7t6pnV6+eWXs2TJkgwaNChJMnz48Gy55ZaZM2dOxo8fnyRpamrK448/nosvvvgdmwsAANi8dOozUgMHDswJJ5yQO++8M6tXr+70L1++fHkWLlyYhQsXJkmeffbZLFy4MIsXL87y5cszbdq0PPjgg3nuuedy33335bDDDsuAAQNy1FFHJUlqa2szYcKEnHXWWbn77rvzm9/8JieccEKGDRtWvosfAABAV+tUSF133XVpb2/PUUcdlYaGhnzuc5/L/PnzC7/OI488kt133z277757kmTq1KnZfffdc/7556dHjx5ZtGhRjjjiiOyyyy456aSTsssuu+TBBx/scGfASy+9NEceeWTGjx+ff/7nf87WW2+dn/70p5vMd0gBAADdT1WpVCp19snLli3LD37wg9x000259957M2TIkJxwwgk5//zzu3LGDa6trS21tbVpbW1N3759Kz3OGnwhL7Cp8YW8xS2+YFilRwDoUjuev6jSI6zV+rZBp65IvammpiannHJKZs+enUcffTTbbLNNvvzlL7+dlwQAAOj23lZIvfbaa7n55ptz5JFHZo899sjLL7+cadOmddVsAAAA3VKn7to3e/bs3HDDDbn11lvTo0ePfPSjH82dd96ZkSNHdvV8AAAA3U6nQurII4/MIYcckmuvvTaHHHJIttxyy66eCwAAoNvqVEg1Nzd3y5syAAAAvBPWO6Ta2to6xFNbW9s614osAABgU7beIbXddtulqakpAwcOzLbbbpuqqqo11pRKpVRVVWXVqlVdOiQAAEB3st4hdc8996Rfv37ln9cWUgAAAJuD9Q6pv74j36hRozbELAAAABuFTn2P1E477ZQvfvGLeeqpp7p6HgAAgG6vUyF1xhln5I477sj73//+DB8+PJdddlmampq6ejYAAIBuqVMhNXXq1MyfPz//8z//k0MPPTRXXHFFdtxxx4wZMybXXXddV88IAADQrXQqpN60yy675Mtf/nKeeuqp3H///XnxxRdzyimndNVsAAAA3VKnvpD3rz388MO58cYb8/3vfz+tra356Ec/2hVzAQAAdFudCqn//d//zQ033JAbb7wxzz33XPbff/989atfzdFHH52ampqunhEAAKBb6VRIve9978uee+6ZSZMm5eMf/3jq6+u7ei4AAIBuq3BIrVq1Kt/61rfy0Y9+tPwFvQAAAJuTwjeb6NGjRz772c+mtbV1Q8wDAADQ7XXqrn3Dhg3L73//+66eBQAAYKPQqZD6t3/7t0ybNi0/+9nP0tTUlLa2tg4bAADApqxTN5s4+OCDkySHH354qqqqyvtLpVKqqqqyatWqrpkOAACgG+pUSN17771dPQcAAMBGo1MhNXLkyK6eAwAAYKPRqZD6xS9+8ZbH99tvv04NAwAAsDHoVEiNGjVqjX1//Vkpn5ECAAA2ZZ26a9/SpUs7bC0tLbnjjjuy1157Zfbs2V09IwAAQLfSqStStbW1a+wbPXp0qqurc+aZZ2bBggVvezAAAIDuqlNXpNZl++23z1NPPdWVLwkAANDtdOqK1GOPPdbhcalUSlNTU7761a9mt91265LBAAAAuqtOhdQHP/jBVFVVpVQqddi/zz775KqrruqSwQAAALqrToXUs88+2+HxFltske233z5bbbVVlwwFAADQnRX6jNSvfvWr/PznP8/gwYPL29y5c7Pffvtlxx13zGmnnZb29vYNNSsAAEC3UCikZsyY0eHzUYsWLcqECRNy0EEH5ZxzzslPf/rTzJw5s8uHBAAA6E4KhdTChQtz4IEHlh/PmjUre++9d6688spMnTo13/jGN3LzzTd3+ZAAAADdSaGQWrp0aerq6sqP586dm4MPPrj8eK+99sqSJUu6bjoAAIBuqFBI1dXVlW80sXLlyvz617/Ohz70ofLxZcuWZcstt+zaCQEAALqZQiF18MEH55xzzsn999+f6dOnZ+utt86HP/zh8vHHHnss7373u7t8SAAAgO6k0O3Pv/KVr+Too4/OyJEj06dPn1x77bXp1atX+fhVV12VMWPGdPmQAAAA3UmhkNp+++1z//33p7W1NX369EmPHj06HP/v//7v9OnTp0sHBAAA6G469YW8tbW1a93fr1+/tzUMAADAxqDQZ6QAAAAQUgAAAIUJKQAAgIKEFAAAQEFCCgAAoCAhBQAAUJCQAgAAKEhIAQAAFCSkAAAAChJSAAAABQkpAACAgoQUAABAQUIKAACgICEFAABQkJACAAAoSEgBAAAUJKQAAAAKElIAAAAFVTSkfvGLX+Swww5LQ0NDqqqqcuutt3Y4XiqVMmPGjDQ0NKR3794ZNWpUnnjiiQ5r2tvbM3ny5AwYMCDbbLNNDj/88Dz//PPv4FkAAACbm4qG1CuvvJLddtstl19++VqPX3zxxbnkkkty+eWXZ/78+amvr8/o0aOzbNmy8popU6bklltuyaxZs/LAAw9k+fLlOfTQQ7Nq1ap36jQAAIDNTM9K/vJx48Zl3Lhxaz1WKpVy2WWX5bzzzsvRRx+dJLn22mtTV1eXG2+8MaeffnpaW1vz3e9+N9dff30OOuigJMn3vve9NDY25q677srYsWPfsXMBAAA2H932M1LPPvtsmpubM2bMmPK+6urqjBw5MvPmzUuSLFiwIK+//nqHNQ0NDRk6dGh5zdq0t7enra2twwYAALC+um1INTc3J0nq6uo67K+rqysfa25uTq9evbLddtutc83azJw5M7W1teWtsbGxi6cHAAA2Zd02pN5UVVXV4XGpVFpj39/6e2umT5+e1tbW8rZkyZIumRUAANg8dNuQqq+vT5I1riy1tLSUr1LV19dn5cqVWbp06TrXrE11dXX69u3bYQMAAFhf3TakhgwZkvr6+syZM6e8b+XKlZk7d25GjBiRJBk+fHi23HLLDmuampry+OOPl9cAAAB0tYretW/58uX53e9+V3787LPPZuHChenXr1923HHHTJkyJRdeeGF23nnn7Lzzzrnwwguz9dZb5/jjj0+S1NbWZsKECTnrrLPSv3//9OvXL9OmTcuwYcPKd/EDAADoahUNqUceeST7779/+fHUqVOTJCeddFKuueaanH322VmxYkUmTpyYpUuXZu+9987s2bNTU1NT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zd1bXYl600YfUkkyderUnHjiidlzzz3zoQ99KN/+9rezePHifPrT/nQEAADoeptESB177LF5+eWXc8EFF6SpqSlDhw7N7bffnsGDB1d6NAAAYBO0SYRUkkycODETJ06s9Biw0aqurs6XvvSlNf7sFYDNh/cCWH9Vpb93Xz8AAAA62Oi/kBcAAOCdJqQAAAAKElIAAAAFCSnYyJRKpZx22mnp169fqqqqsnDhworM8dxzz1X09wPwzjn55JNz5JFHVnoM6FY2mbv2webijjvuyDXXXJP77rsvO+20UwYMGFDpkQAANjtCCjYyzzzzTAYNGpQRI0ZUehQAgM2WP+2DjcjJJ5+cyZMnZ/Hixamqqsq73vWulEqlXHzxxdlpp53Su3fv7LbbbvnBD35Qfs59992Xqqqq3Hnnndl9993Tu3fvHHDAAWlpacnPf/7zvP/970/fvn1z3HHH5dVXXy0/74477si+++6bbbfdNv3798+hhx6aZ5555i3n++1vf5uPfOQj6dOnT+rq6nLiiSfmpZde2mD/PQBY06hRozJ58uRMmTIl2223Xerq6vLtb387r7zySk455ZTU1NTk3e9+d37+858nSVatWpUJEyZkyJAh6d27d9773vfm61//+lv+jr/33gObAyEFG5Gvf/3rueCCC7LDDjukqakp8+fPz7/8y7/k6quvzhVXXJEnnngiZ555Zk444YTMnTu3w3NnzJiRyy+/PPPmzcuSJUsyfvz4XHbZZbnxxhtz2223Zc6cOfnmN79ZXv/KK69k6tSpmT9/fu6+++5sscUWOeqoo7J69eq1ztbU1JSRI0fmgx/8YB555JHccccd+eMf/5jx48dv0P8mAKzp2muvzYABA/Lwww9n8uTJ+cxnPpOPfexjGTFiRH79619n7NixOfHEE/Pqq69m9erV2WGHHXLzzTfnt7/9bc4///yce+65ufnmm9f5+uv73gObtBKwUbn00ktLgwcPLpVKpdLy5ctLW221VWnevHkd1kyYMKF03HHHlUqlUunee+8tJSnddddd5eMzZ84sJSk988wz5X2nn356aezYsev8vS0tLaUkpUWLFpVKpVLp2WefLSUp/eY3vymVSqXSF7/4xdKYMWM6PGfJkiWlJKWnnnqq0+cLQDEjR44s7bvvvuXHb7zxRmmbbbYpnXjiieV9TU1NpSSlBx98cK2vMXHixNIxxxxTfnzSSSeVjjjiiFKptH7vPbA58Bkp2Ij99re/zWuvvZbRo0d32L9y5crsvvvuHfbtuuuu5Z/r6uqy9dZbZ6edduqw7+GHHy4/fuaZZ/LFL34xDz30UF566aXylajFixdn6NCha8yyYMGC3HvvvenTp88ax5555pnssssunTtJAAr763/ze/Tokf79+2fYsGHlfXV1dUmSlpaWJMm3vvWtfOc738kf/vCHrFixIitXrswHP/jBtb52kfce2JQJKdiIvRk3t912W/7hH/6hw7Hq6uoOj7fccsvyz1VVVR0ev7nvr/9s77DDDktjY2OuvPLKNDQ0ZPXq1Rk6dGhWrly5zlkOO+ywXHTRRWscGzRoULETA+BtWdu/8X/7PpD85d/um2++OWeeeWa+9rWv5UMf+lBqamry7//+7/nVr3611tcu8t4DmzIhBRuxD3zgA6murs7ixYszcuTILnvdl19+OU8++WT+67/+Kx/+8IeTJA888MBbPmePPfbID3/4w7zrXe9Kz57+aQHYWNx///0ZMWJEJk6cWN73VjcX2lDvPbCx8b92YCNWU1OTadOm5cwzz8zq1auz7777pq2tLfPmzUufPn1y0kkndep1t9tuu/Tv3z/f/va3M2jQoCxevDjnnHPOWz5n0qRJufLKK3Pcccfl85//fAYMGJDf/e53mTVrVq688sr06NGjU7MAsGG95z3vyXXXXZc777wzQ4YMyfXXX5/58+dnyJAha12/od57YGMjpGAj96//+q8ZOHBgZs6cmd///vfZdttts8cee+Tcc8/t9GtuscUWmTVrVj772c9m6NChee9735tvfOMbGTVq1Dqf09DQkF/+8pf5whe+kLFjx6a9vT2DBw/OwQcfnC22cINQgO7q05/+dBYuXJhjjz02VVVVOe644zJx4sTy7dHXZkO898DGpqpUKpUqPQQAAMDGxP9NDAAAUJCQAgAAKEhIAQAAFCSkAAAAChJSAAAABQkpAACAgoQUAABAQUIKAACgICEFwCavpaUlp59+enbcccdUV1envr4+Y8eOzYMPPljp0QDYSPWs9AAAsKEdc8wxef3113Pttddmp512yh//+Mfcfffd+dOf/lTp0QDYSLkiBcAm7c9//nMeeOCBXHTRRdl///0zePDg/NM//VOmT5+eQw45JEnS2tqa0047LQMHDkzfvn1zwAEH5NFHH02SvPjii6mvr8+FF15Yfs1f/epX6dWrV2bPnl2RcwKg8oQUAJu0Pn36pE+fPrn11lvT3t6+xvFSqZRDDjkkzc3Nuf3227NgwYLsscceOfDAA/OnP/0p22+/fa666qrMmDEjjzzySJYvX54TTjghEydOzJgxYypwRgB0B1WlUqlU6SEAYEP64Q9/mE996lNZsWJF9thjj4wcOTIf//jHs+uuu+aee+7JUUcdlZaWllRXV5ef8573vCdnn312TjvttCTJpEmTctddd2WvvfbKo48+mvnz52errbaq1CkBUGFCCoDNwmuvvZb7778/Dz74YO644448/PDD+c53vpMXX3wx55xzTnr37t1h/YoVKzJt2rRcdNFF5cdDhw7NkiVL8sgjj2TXXXetxGkA0E0IKQA2S5/85CczZ86cTJw4Md/85jdz3333rbFm2223zYABA5IkTzzxRPbcc8+8/vrrueWWW3LYYYe9wxMD0J24ax8Am6UPfOADufXWW7PHHnukubk5PXv2zLve9a61rl25cmU+8YlP5Nhjj8373ve+TJgwIYsWLUpdXd07OzQA3YYrUgBs0l5++eV87GMfy6mnnppdd901NTU1eeSRRzJ58uQccsgh+c53vpP99tsvy5Yty0UXXZT3vve9eeGFF3L77bfnyCOPzJ577pnPf/7z+cEPfpBHH300ffr0yf7775+ampr87Gc/q/TpAVAhQgqATVp7e3tmzJiR2bNn55lnnsnrr7+exsbGfOxjH8u5556b3r17Z9myZTnvvPPywx/+sHy78/322y8zZ87MM888k9GjR+fee+/NvvvumyRZvHhxdt1118ycOTOf+cxnKnyGAFSCkAIAACjI90gBAAAUJKQAAAAKElIAAAAFCSkAAICChBQAAEBBQgoAAKAgIQUAAFCQkAIAAChISAEAABQkpAAAAAoSUgAAAAUJKQAAgIL+P4+/MZxGsklUAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 1000x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "prdata=df.groupby('Sex').agg({'Survived':'count'}).reset_index()\n",
    "fig, (ax1) = plt.subplots(1,1,figsize=(10, 5))\n",
    "sns.barplot(x='Sex', y='Survived', data = prdata, ax=ax1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "f6842d95",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:46.713436Z",
     "iopub.status.busy": "2023-08-08T14:50:46.713000Z",
     "iopub.status.idle": "2023-08-08T14:50:46.966404Z",
     "shell.execute_reply": "2023-08-08T14:50:46.965128Z"
    },
    "papermill": {
     "duration": 0.279745,
     "end_time": "2023-08-08T14:50:46.969312",
     "exception": false,
     "start_time": "2023-08-08T14:50:46.689567",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Pclass', ylabel='Survived'>"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA1IAAAHACAYAAACoF1lmAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAAAj9klEQVR4nO3df5SWdZ3/8dfND0fUmVFQZpjjwGJhW4F2QvM3QgjJpqbkamu7XzVq20UpQr6atd8N25KyXXHT1jZP+asQcwv7sWaQJv5aC2lJc11WiwKTCTOaAcJBcb5/uM5pVIzPNHDPDI/HOdc53Nf1ue/7fXVOc3ye676vu9LR0dERAAAAdtiAag8AAADQ1wgpAACAQkIKAACgkJACAAAoJKQAAAAKCSkAAIBCQgoAAKCQkAIAACg0qNoD9AbPP/98nnzyydTW1qZSqVR7HAAAoEo6OjqycePGNDU1ZcCA7V93ElJJnnzyyTQ3N1d7DAAAoJdYu3ZtDjzwwO0eF1JJamtrk7zwP1ZdXV2VpwEAAKqlra0tzc3NnY2wPUIq6fw4X11dnZACAAD+4Fd+3GwCAACgkJACAAAoJKQAAAAKCSkAAIBCQgoAAKCQkAIAACgkpAAAAAoJKQAAgEJCCgAAoJCQAgAAKCSkAAAACgkpAACAQkIKAACgkJACAAAoJKQAAAAKDar2AAAAu5tjrjym2iNAn3TfrPuqPUInV6QAAAAKCSkAAIBCQgoAAKCQkAIAACgkpAAAAAoJKQAAgEJCCgAAoJCQAgAAKCSkAAAACgkpAACAQkIKAACgkJACAAAoJKQAAAAKCSkAAIBCQgoAAKCQkAIAACgkpAAAAAoJKQAAgEJCCgAAoJCQAgAAKCSkAAAACgkpAACAQkIKAACgkJACAAAoJKQAAAAKVTWk5s+fn8MPPzy1tbUZPnx4Tj311KxatarLmo6OjsybNy9NTU0ZMmRIJk6cmEceeaTLmvb29syaNSv7779/9t5775xyyil54oknduWpAAAAu5GqhtSyZcty3nnn5YEHHsjSpUvz3HPPZerUqdm8eXPnmssuuyyXX355rrrqqixfvjyNjY2ZMmVKNm7c2Llm9uzZWbx4cRYtWpR77703mzZtykknnZRt27ZV47QAAIB+rtLR0dFR7SFe9NRTT2X48OFZtmxZJkyYkI6OjjQ1NWX27Nm56KKLkrxw9amhoSGf/vSn8/73vz+tra054IADcuONN+bMM89Mkjz55JNpbm7Obbfdlre97W1/8H3b2tpSX1+f1tbW1NXV7dRzBAA45spjqj0C9En3zbpvp7/HjrZBr/qOVGtra5Jk6NChSZLVq1enpaUlU6dO7VxTU1OT448/Pvfff3+SZMWKFXn22We7rGlqasrYsWM717xUe3t72traumwAAAA7qteEVEdHR+bMmZNjjz02Y8eOTZK0tLQkSRoaGrqsbWho6DzW0tKSPfbYI/vtt99217zU/PnzU19f37k1Nzf39OkAAAD9WK8JqfPPPz8PPfRQbrrpppcdq1QqXR53dHS8bN9Lvdqaiy++OK2trZ3b2rVruz84AACw2+kVITVr1qx885vfzPe///0ceOCBnfsbGxuT5GVXltavX995laqxsTFbt27Nhg0btrvmpWpqalJXV9dlAwAA2FFVDamOjo6cf/75+frXv54777wzo0eP7nJ89OjRaWxszNKlSzv3bd26NcuWLcvRRx+dJBk/fnwGDx7cZc26devyk5/8pHMNAABATxpUzTc/77zzsnDhwnzjG99IbW1t55Wn+vr6DBkyJJVKJbNnz86ll16aMWPGZMyYMbn00kuz11575ayzzupcO2PGjFxwwQUZNmxYhg4dmrlz52bcuHE54YQTqnl6AABAP1XVkLr66quTJBMnTuyy/9prr80555yTJLnwwguzZcuWzJw5Mxs2bMgRRxyRJUuWpLa2tnP9ggULMmjQoJxxxhnZsmVLJk+enOuuuy4DBw7cVacCAADsRnrV70hVi9+RAgB2Jb8jBd3jd6QAAAD6MCEFAABQSEgBAAAUElIAAACFhBQAAEAhIQUAAFBISAEAABQSUgAAAIWEFAAAQCEhBQAAUEhIAQAAFBJSAAAAhYQUAABAISEFAABQSEgBAAAUElIAAACFhBQAAEAhIQUAAFBISAEAABQSUgAAAIWEFAAAQCEhBQAAUEhIAQAAFBJSAAAAhYQUAABAISEFAABQSEgBAAAUElIAAACFhBQAAEAhIQUAAFBISAEAABQSUgAAAIWEFAAAQCEhBQAAUEhIAQAAFBJSAAAAhYQUAABAISEFAABQSEgBAAAUElIAAACFhBQAAEAhIQUAAFBISAEAABQSUgAAAIWEFAAAQCEhBQAAUEhIAQAAFBJSAAAAhYQUAABAISEFAABQSEgBAAAUElIAAACFhBQAAEAhIQUAAFBISAEAABQSUgAAAIWEFAAAQCEhBQAAUEhIAQAAFBJSAAAAhYQUAABAISEFAABQSEgBAAAUElIAAACFhBQAAEAhIQUAAFBISAEAABQSUgAAAIWEFAAAQCEhBQAAUEhIAQAAFBJSAAAAhYQUAABAoaqG1N13352TTz45TU1NqVQqufXWW7scP+ecc1KpVLpsRx55ZJc17e3tmTVrVvbff//svffeOeWUU/LEE0/swrMAAAB2N1UNqc2bN+fQQw/NVVddtd01J554YtatW9e53XbbbV2Oz549O4sXL86iRYty7733ZtOmTTnppJOybdu2nT0+AACwmxpUzTefNm1apk2b9qprampq0tjY+IrHWltb88UvfjE33nhjTjjhhCTJl7/85TQ3N+d73/te3va2t/X4zAAAAL3+O1J33XVXhg8fnoMPPjjve9/7sn79+s5jK1asyLPPPpupU6d27mtqasrYsWNz//33b/c129vb09bW1mUDAADYUb06pKZNm5avfOUrufPOO/NP//RPWb58ed761remvb09SdLS0pI99tgj++23X5fnNTQ0pKWlZbuvO3/+/NTX13duzc3NO/U8AACA/qWqH+37Q84888zOf48dOzaHHXZYRo0alX//93/P9OnTt/u8jo6OVCqV7R6/+OKLM2fOnM7HbW1tYgoAANhhvfqK1EuNGDEio0aNymOPPZYkaWxszNatW7Nhw4Yu69avX5+Ghobtvk5NTU3q6uq6bAAAADuqT4XU008/nbVr12bEiBFJkvHjx2fw4MFZunRp55p169blJz/5SY4++uhqjQkAAPRzVf1o36ZNm/L44493Pl69enVWrlyZoUOHZujQoZk3b17e+c53ZsSIEfn5z3+ej3zkI9l///1z2mmnJUnq6+szY8aMXHDBBRk2bFiGDh2auXPnZty4cZ138QMAAOhpVQ2pBx98MJMmTep8/OL3ls4+++xcffXVefjhh3PDDTfkt7/9bUaMGJFJkybl5ptvTm1tbedzFixYkEGDBuWMM87Ili1bMnny5Fx33XUZOHDgLj8fAABg91Dp6OjoqPYQ1dbW1pb6+vq0trb6vhQAsNMdc+Ux1R4B+qT7Zt23099jR9ugT31HCgAAoDcQUgAAAIWEFAAAQCEhBQAAUEhIAQAAFBJSAAAAhYQUAABAISEFAABQSEgBAAAUElIAAACFhBQAAEAhIQUAAFBISAEAABQSUgAAAIWEFAAAQCEhBQAAUEhIAQAAFBJSAAAAhYQUAABAISEFAABQSEgBAAAUElIAAACFhBQAAEAhIQUAAFBISAEAABQSUgAAAIUG7ejC6dOn7/CLfv3rX+/WMAAAAH3BDl+Rqq+v79zq6upyxx135MEHH+w8vmLFitxxxx2pr6/fKYMCAAD0Fjt8Reraa6/t/PdFF12UM844I5///OczcODAJMm2bdsyc+bM1NXV9fyUAAAAvUi3viP1pS99KXPnzu2MqCQZOHBg5syZky996Us9NhwAAEBv1K2Qeu655/Loo4++bP+jjz6a559//o8eCgAAoDfb4Y/2/b5zzz0373nPe/L444/nyCOPTJI88MAD+dSnPpVzzz23RwcEAADobboVUv/4j/+YxsbGLFiwIOvWrUuSjBgxIhdeeGEuuOCCHh0QAACgt+lWSA0YMCAXXnhhLrzwwrS1tSWJm0wAAAC7jW7/IO9zzz2X733ve7nppptSqVSSJE8++WQ2bdrUY8MBAAD0Rt26IvWLX/wiJ554YtasWZP29vZMmTIltbW1ueyyy/LMM8/k85//fE/PCQAA0Gt064rUBz/4wRx22GHZsGFDhgwZ0rn/tNNOyx133NFjwwEAAPRG3boide+99+a+++7LHnvs0WX/qFGj8stf/rJHBgMAAOitunVF6vnnn8+2bdtetv+JJ55IbW3tHz0UAABAb9atkJoyZUquuOKKzseVSiWbNm3Kxz72sfzZn/1ZT80GAADQK3Xro30LFizIpEmT8oY3vCHPPPNMzjrrrDz22GPZf//9c9NNN/X0jAAAAL1Kt0KqqakpK1euzE033ZQf/ehHef755zNjxoy8+93v7nLzCQAAgP6oWyH1u9/9LnvttVfe85735D3veU9PzwQAANCrdes7UsOHD89f/uVf5rvf/W6ef/75np4JAACgV+tWSN1www1pb2/Paaedlqampnzwgx/M8uXLe3o2AACAXqlbITV9+vTccsst+dWvfpX58+fn0UcfzdFHH52DDz44H//4x3t6RgAAgF6lWyH1otra2px77rlZsmRJfvzjH2fvvffOJZdc0lOzAQAA9Ep/VEg988wz+epXv5pTTz01b37zm/P0009n7ty5PTUbAABAr9Stu/YtWbIkX/nKV3Lrrbdm4MCBOf300/Pd7343xx9/fE/PBwAA0Ot0K6ROPfXUvP3tb8/111+ft7/97Rk8eHBPzwUAANBrdSukWlpaUldX19OzAAAA9Ak7HFJtbW1d4qmtrW27a0UWAADQn+1wSO23335Zt25dhg8fnn333TeVSuVlazo6OlKpVLJt27YeHRIAAKA32eGQuvPOOzN06NDOf79SSAEAAOwOdjikfv+OfBMnTtwZswAAAPQJ3fodqYMOOij/7//9v6xataqn5wEAAOj1uhVS559/fm6//fa8/vWvz/jx43PFFVdk3bp1PT0bAABAr9StkJozZ06WL1+e//7v/85JJ52Uq6++OiNHjszUqVNzww039PSMAAAAvUq3QupFBx98cC655JKsWrUq99xzT5566qmce+65PTUbAABAr9StH+T9fT/84Q+zcOHC3HzzzWltbc3pp5/eE3MBAAD0Wt0Kqf/5n//JV77ylSxcuDA///nPM2nSpHzqU5/K9OnTU1tb29MzAgAA9CrdCqk//dM/zWGHHZbzzjsv73rXu9LY2NjTcwEAAPRaxSG1bdu2fP7zn8/pp5/e+QO9AAAAu5Pim00MHDgwH/jAB9La2roz5gEAAOj1unXXvnHjxuVnP/tZT88CAADQJ3QrpD75yU9m7ty5+fa3v51169alra2tywYAANCfdetmEyeeeGKS5JRTTkmlUunc39HRkUqlkm3btvXMdP3Q+P/rB4uh1IrP/J9qjwAA0EW3Qur73/9+T88BAADQZ3QrpI4//viengMAAKDP6FZI3X333a96fMKECd0aBgAAoC/oVkhNnDjxZft+/7tSviMFAAD0Z926a9+GDRu6bOvXr8/tt9+eww8/PEuWLOnpGQEAAHqVbl2Rqq+vf9m+KVOmpKamJh/60IeyYsWKP3owAACA3qpbV6S254ADDsiqVat2eP3dd9+dk08+OU1NTalUKrn11lu7HO/o6Mi8efPS1NSUIUOGZOLEiXnkkUe6rGlvb8+sWbOy//77Z++9984pp5ySJ554oidOBwAA4BV1K6QeeuihLtuPf/zj3H777fnbv/3bHHrooTv8Ops3b86hhx6aq6666hWPX3bZZbn88stz1VVXZfny5WlsbMyUKVOycePGzjWzZ8/O4sWLs2jRotx7773ZtGlTTjrpJN/TAgAA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tcMAVTFjxozccccdWbduXerr63PIIYfkoosuypQpU6o9Gr2YkAIAACg0oNoDAAAA9DVCCgAAoJCQAgAAKCSkAAAACgkpAACAQkIKAACgkJACAAAoJKQA2C2dc845OfXUU6s9BgB9lJACoM8655xzUqlUUqlUMnjw4Bx00EGZO3duNm/eXO3RAOjnBlV7AAD4Y5x44om59tpr8+yzz+aee+7Je9/73mzevDlXX311tUcDoB9zRQqAPq2mpiaNjY1pbm7OWWedlXe/+9259dZbkySPPPJI3v72t6euri61tbU57rjj8tOf/vQVX+f222/Psccem3333TfDhg3LSSed1GXt1q1bc/7552fEiBHZc8898yd/8ieZP39+5/F58+Zl5MiRqampSVNTUz7wgQ/s1PMGoLpckQKgXxkyZEieffbZ/PKXv8yECRMyceLE3Hnnnamrq8t9992X55577hWft3nz5syZMyfjxo3L5s2b8/d///c57bTTsnLlygwYMCCf/exn881vfjNf/epXM3LkyKxduzZr165Nkvzbv/1bFixYkEWLFuWNb3xjWlpa8uMf/3hXnjYAu5iQAqDf+OEPf5iFCxdm8uTJ+dznPpf6+vosWrQogwcPTpIcfPDB233uO9/5zi6Pv/jFL2b48OH5r//6r4wdOzZr1qzJmDFjcuyxx6ZSqWTUqFGda9esWZPGxsaccMIJGTx4cEaOHJm3vOUtO+ckAegVfLQPgD7t29/+dvbZZ5/sueeeOeqoozJhwoRceeWVWblyZY477rjOiPpDfvrTn+ass87KQQcdlLq6uowePTrJC5GUvHBji5UrV+Z1r3tdPvCBD2TJkiWdz/3zP//zbNmyJQcddFDe9773ZfHixdu98gVA/yCkAOjTJk2alJUrV2bVqlV55pln8vWvfz3Dhw/PkCFDil7n5JNPztNPP51rrrkmP/jBD/KDH/wgyQvfjUqSN7/5zVm9enX+4R/+IVu2bMkZZ5yR008/PUnS3NycVatW5XOf+1yGDBmSmTNnZsKECXn22Wd79mQB6DWEFAB92t57753Xvva1GTVqVJerT4ccckjuueeeHYqZp59+Oo8++mj+7u/+LpMnT87rX//6bNiw4WXr6urqcuaZZ+aaa67JzTffnK997Wv5zW9+k+SF72adcsop+exnP5u77ror//Ef/5GHH364504UgF7Fd6QA6JfOP//8XHnllXnXu96Viy++OPX19XnggQfylre8Ja973eu6rN1vv/0ybNiwfOELX8iIESOyZs2afPjDH+6yZsGCBRkxYkTe9KY3ZcCAAbnlllvS2NiYfffdN9ddd122bduWI444InvttVduvPHGDBkypMv3qADoX1yRAqBfGjZsWO68885s2rQpxx9/fMaPH59rrrnmFb8zNWDAgCxatCgrVqzI2LFj86EPfSif+cxnuqzZZ5998ulPfzqHHXZYDj/88Pz85z/PbbfdlgEDBmTffffNNddck2OOOSaHHHJI7rjjjnzrW9/KsGHDdtXpArCLVTo6OjqqPQQAAEBf4ooUAABAISEFAABQSEgBAAAUElIAAACFhBQAAEAhIQUAAFBISAEAABQSUgAAAIWEFAAAQCEhBQAAUEhIAQAAFBJSAAAAhf4/pkp36FCJzL4AAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 1000x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "prdata=df.groupby('Pclass').agg({'Survived':'count'}).reset_index()\n",
    "fig, (ax1) = plt.subplots(1,1,figsize=(10, 5))\n",
    "sns.barplot(x='Pclass', y='Survived', data = prdata, ax=ax1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "3f318c61",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:47.013452Z",
     "iopub.status.busy": "2023-08-08T14:50:47.013046Z",
     "iopub.status.idle": "2023-08-08T14:50:47.270720Z",
     "shell.execute_reply": "2023-08-08T14:50:47.269431Z"
    },
    "papermill": {
     "duration": 0.282774,
     "end_time": "2023-08-08T14:50:47.273216",
     "exception": false,
     "start_time": "2023-08-08T14:50:46.990442",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Embarked', ylabel='Survived'>"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA1IAAAHACAYAAACoF1lmAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAAApiElEQVR4nO3df5TWdZ3//8cIMgLCKAIzzDpyKHHVhaxFV9FW8Ae//JlY2molSq17RIyQLLKSSsX8uOopV2xdFfyBWqcyLUXwB/h7U5L1R4iauOI6E6Y4A4aDwvX9Y79eZyegeE8DM8Dtds77xPV+v67rel6c03W8876u91VRKpVKAQAAYJPt0N4DAAAAbG2EFAAAQEFCCgAAoCAhBQAAUJCQAgAAKEhIAQAAFCSkAAAAChJSAAAABXVu7wE6gnXr1uWNN95Ijx49UlFR0d7jAAAA7aRUKmXlypWpra3NDjts/LyTkEryxhtvpK6urr3HAAAAOohly5Zl99133+hxIZWkR48eSf73L6tnz57tPA0AANBempqaUldXV26EjRFSSfnjfD179hRSAADAX/zKj4tNAAAAFCSkAAAAChJSAAAABQkpAACAgoQUAABAQUIKAACgICEFAABQkJACAAAoSEgBAAAUJKQAAAAKElIAAAAFCSkAAICChBQAAEBBQgoAAKAgIQUAAFBQ5/YeAABge3PIDw9p7xFgq/ToxEfbe4QyZ6QAAAAKElIAAAAFCSkAAICChBQAAEBBQgoAAKAgIQUAAFCQkAIAAChISAEAABQkpAAAAAoSUgAAAAUJKQAAgIKEFAAAQEFCCgAAoCAhBQAAUJCQAgAAKEhIAQAAFCSkAAAAChJSAAAABQkpAACAgoQUAABAQUIKAACgICEFAABQULuG1PTp03PAAQekR48e6du3bz71qU9lyZIlLdaMGzcuFRUVLbaDDjqoxZrm5uZMnDgxvXv3Tvfu3XPcccfl9ddf35IvBQAA2I60a0gtWLAgEyZMyBNPPJF58+blgw8+yMiRI/Puu++2WDd69OjU19eXt7vvvrvF8UmTJuXnP/95brvttjzyyCNZtWpVjjnmmKxdu3ZLvhwAAGA70bk9n3zOnDktbt9www3p27dvFi5cmEMPPbS8v7KyMjU1NRt8jMbGxlx33XW56aabcuSRRyZJbr755tTV1eW+++7LqFGjNt8LAAAAtksd6jtSjY2NSZJevXq12D9//vz07ds3e+21V770pS9l+fLl5WMLFy7M+++/n5EjR5b31dbWZtCgQXnsscc2+DzNzc1pampqsQEAAGyqDhNSpVIpkydPzic/+ckMGjSovH/MmDG55ZZb8sADD+Rf//Vf8+STT+bwww9Pc3NzkqShoSFdunTJrrvu2uLxqqur09DQsMHnmj59eqqqqspbXV3d5nthAADANqddP9r3f5199tl55pln8sgjj7TYf/LJJ5f/PGjQoOy///7p379/fvWrX2Xs2LEbfbxSqZSKiooNHps6dWomT55cvt3U1CSmAACATdYhzkhNnDgxd955Zx588MHsvvvuf3Ztv3790r9//7z00ktJkpqamqxZsyYrVqxosW758uWprq7e4GNUVlamZ8+eLTYAAIBN1a4hVSqVcvbZZ+dnP/tZHnjggQwYMOAv3uett97KsmXL0q9fvyTJkCFDsuOOO2bevHnlNfX19Xnuuedy8MEHb7bZAQCA7Ve7frRvwoQJmT17dn7xi1+kR48e5e80VVVVpWvXrlm1alWmTZuWE088Mf369curr76ab3zjG+ndu3dOOOGE8trx48fn3HPPzW677ZZevXplypQpGTx4cPkqfgAAAG2pXUNqxowZSZLhw4e32H/DDTdk3Lhx6dSpU5599tnceOONeeedd9KvX78cdthhuf3229OjR4/y+iuuuCKdO3fOSSedlNWrV+eII47IzJkz06lTpy35cgAAgO1ERalUKrX3EO2tqakpVVVVaWxs9H0pAGCzO+SHh7T3CLBVenTio5v9OTa1DTrExSYAAAC2JkIKAACgICEFAABQkJACAAAoSEgBAAAUJKQAAAAKElIAAAAFCSkAAICChBQAAEBBQgoAAKAgIQUAAFCQkAIAAChISAEAABQkpAAAAAoSUgAAAAUJKQAAgIKEFAAAQEFCCgAAoCAhBQAAUJCQAgAAKEhIAQAAFCSkAAAAChJSAAAABQkpAACAgoQUAABAQUIKAACgICEFAABQkJACAAAoSEgBAAAUJKQAAAAKElIAAAAFCSkAAICChBQAAEBBQgoAAKAgIQUAAFCQkAIAAChISAEAABQkpAAAAAoSUgAAAAUJKQAAgIKEFAAAQEFCCgAAoCAhBQAAUJCQAgAAKEhIAQAAFCSkAAAAChJSAAAABQkpAACAgoQUAABAQUIKAACgICEFAABQkJACAAAoSEgBAAAUJKQAAAAKElIAAAAFCSkAAICChBQAAEBB7RpS06dPzwEHHJAePXqkb9+++dSnPpUlS5a0WFMqlTJt2rTU1tama9euGT58eJ5//vkWa5qbmzNx4sT07t073bt3z3HHHZfXX399S74UAABgO9KuIbVgwYJMmDAhTzzxRObNm5cPPvggI0eOzLvvvltec+mll+byyy/PVVddlSeffDI1NTUZMWJEVq5cWV4zadKk/PznP89tt92WRx55JKtWrcoxxxyTtWvXtsfLAgAAtnEVpVKp1N5DfOjNN99M3759s2DBghx66KEplUqpra3NpEmT8rWvfS3J/559qq6uzve///2ceeaZaWxsTJ8+fXLTTTfl5JNPTpK88cYbqaury913351Ro0b9xedtampKVVVVGhsb07Nnz836GgEADvnhIe09AmyVHp346GZ/jk1tgw71HanGxsYkSa9evZIkS5cuTUNDQ0aOHFleU1lZmWHDhuWxxx5LkixcuDDvv/9+izW1tbUZNGhQec2fam5uTlNTU4sNAABgU3WYkCqVSpk8eXI++clPZtCgQUmShoaGJEl1dXWLtdXV1eVjDQ0N6dKlS3bdddeNrvlT06dPT1VVVXmrq6tr65cDAABswzpMSJ199tl55plncuutt653rKKiosXtUqm03r4/9efWTJ06NY2NjeVt2bJlrR8cAADY7nSIkJo4cWLuvPPOPPjgg9l9993L+2tqapJkvTNLy5cvL5+lqqmpyZo1a7JixYqNrvlTlZWV6dmzZ4sNAABgU7VrSJVKpZx99tn52c9+lgceeCADBgxocXzAgAGpqanJvHnzyvvWrFmTBQsW5OCDD06SDBkyJDvuuGOLNfX19XnuuefKawAAANpS5/Z88gkTJmT27Nn5xS9+kR49epTPPFVVVaVr166pqKjIpEmTcvHFF2fgwIEZOHBgLr744nTr1i2nnHJKee348eNz7rnnZrfddkuvXr0yZcqUDB48OEceeWR7vjwAAGAb1a4hNWPGjCTJ8OHDW+y/4YYbMm7cuCTJeeedl9WrV+ess87KihUrcuCBB2bu3Lnp0aNHef0VV1yRzp0756STTsrq1atzxBFHZObMmenUqdOWeikAAMB2pEP9jlR78TtSAMCW5HekoHX8jhQAAMBWTEgBAAAUJKQAAAAKElIAAAAFCSkAAICChBQAAEBBQgoAAKAgIQUAAFCQkAIAAChISAEAABQkpAAAAAoSUgAAAAUJKQAAgIKEFAAAQEFCCgAAoCAhBQAAUJCQAgAAKEhIAQAAFCSkAAAAChJSAAAABQkpAACAgoQUAABAQUIKAACgICEFAABQkJACAAAoSEgBAAAUJKQAAAAKElIAAAAFCSkAAICChBQAAEBBQgoAAKAgIQUAAFCQkAIAAChISAEAABQkpAAAAAoSUgAAAAUJKQAAgIKEFAAAQEFCCgAAoCAhBQAAUJCQAgAAKEhIAQAAFCSkAAAAChJSAAAABQkpAACAgjpv6sKxY8du8oP+7Gc/a9UwAAAAW4NNPiNVVVVV3nr27Jn7778/Tz31VPn4woULc//996eqqmqzDAoAANBRbPIZqRtuuKH856997Ws56aSTcs0116RTp05JkrVr1+ass85Kz549235KAACADqRV35G6/vrrM2XKlHJEJUmnTp0yefLkXH/99W02HAAAQEfUqpD64IMPsnjx4vX2L168OOvWrfurhwIAAOjINvmjff/X6aefnjPOOCMvv/xyDjrooCTJE088kUsuuSSnn356mw4IAADQ0bQqpC677LLU1NTkiiuuSH19fZKkX79+Oe+883Luuee26YAAAAAdTatCaocddsh5552X8847L01NTUniIhMAAMB2o9U/yPvBBx/kvvvuy6233pqKiookyRtvvJFVq1a12XAAAAAdUavOSP33f/93Ro8enddeey3Nzc0ZMWJEevTokUsvvTTvvfderrnmmraeEwAAoMNo1RmpL3/5y9l///2zYsWKdO3atbz/hBNOyP33399mwwEAAHRErToj9cgjj+TRRx9Nly5dWuzv379//ud//qdNBgMAAOioWnVGat26dVm7du16+19//fX06NFjkx/noYceyrHHHpva2tpUVFTkjjvuaHF83LhxqaioaLF9eLn1DzU3N2fixInp3bt3unfvnuOOOy6vv/56a14WAADAJmlVSI0YMSJXXnll+XZFRUVWrVqVCy64IEcdddQmP867776b/fbbL1ddddVG14wePTr19fXl7e67725xfNKkSfn5z3+e2267LY888khWrVqVY445ZoOhBwAA0BZa9dG+K664Iocddlj23XffvPfeeznllFPy0ksvpXfv3rn11ls3+XHGjBmTMWPG/Nk1lZWVqamp2eCxxsbGXHfddbnpppty5JFHJkluvvnm1NXV5b777suoUaM2/UUBAABsolaFVG1tbRYtWpRbb701v/nNb7Ju3bqMHz8+p556aouLT7SF+fPnp2/fvtlll10ybNiwXHTRRenbt2+SZOHChXn//fczcuTIFrMNGjQojz322EZDqrm5Oc3NzeXbH/4WFgAAwKZoVUj98Y9/TLdu3XLGGWfkjDPOaOuZysaMGZPPfOYz6d+/f5YuXZpvfetbOfzww7Nw4cJUVlamoaEhXbp0ya677triftXV1WloaNjo406fPj3f+c53NtvcAADAtq1V35Hq27dvPve5z+Xee+/NunXr2nqmspNPPjlHH310Bg0alGOPPTb33HNPXnzxxfzqV7/6s/crlUrlHwnekKlTp6axsbG8LVu2rK1HBwAAtmGtCqkbb7wxzc3NOeGEE1JbW5svf/nLefLJJ9t6tvX069cv/fv3z0svvZQkqampyZo1a7JixYoW65YvX57q6uqNPk5lZWV69uzZYgMAANhUrQqpsWPH5ic/+Ul+//vfZ/r06Vm8eHEOPvjg7LXXXvnud7/b1jOWvfXWW1m2bFn69euXJBkyZEh23HHHzJs3r7ymvr4+zz33XA4++ODNNgcAALB9a1VIfahHjx45/fTTM3fu3PzXf/1XunfvXui7R6tWrcqiRYuyaNGiJMnSpUuzaNGivPbaa1m1alWmTJmSxx9/PK+++mrmz5+fY489Nr17984JJ5yQJKmqqsr48eNz7rnn5v7778/TTz+dz33ucxk8eHD5Kn4AAABtrVUXm/jQe++9lzvvvDOzZ8/OnDlz0rdv30yZMmWT7//UU0/lsMMOK9+ePHlykuS0007LjBkz8uyzz+bGG2/MO++8k379+uWwww7L7bff3uJHf6+44op07tw5J510UlavXp0jjjgiM2fOTKdOnf6alwYAALBRFaVSqVT0TnPnzs0tt9ySO+64I506dcqnP/3pnHrqqRk2bNjmmHGza2pqSlVVVRobG31fCgDY7A754SHtPQJslR6d+Ohmf45NbYNWnZH61Kc+laOPPjqzZs3K0UcfnR133LHVgwIAAGxtWhVSDQ0NztwAAADbrU0Oqaamphbx1NTUtNG1IgsAANiWbXJI7brrrqmvr0/fvn2zyy67bPAHbz/8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V1yRzp0756STTsrq1atzxBFHZObMmR3yN6QAAIBtQ0WpVCq19xDtrampKVVVVWlsbNzs35fyg7xQnB/kBbY1h/zwkPYeAbZKj058dLM/x6a2QYf9jhQAAEBHJaQAAAAKElIAAAAFCSkAAICChBQAAEBBQgoAAKAgIQUAAFCQkAIAAChISAEAABQkpAAAAAoSUgAAAAUJKQAAgIKEFAAAQEFCCgAAoCAhBQAAUJCQAgAAKEhIAQAAFCSkAAAAChJSAAAABQkpAACAgoQUAABAQUIKAACgICEFAABQkJACAAAoSEgBAAAUJKQAAAAKElIAAAAFCSkAAICChBQAAEBBQgoAAKAgIQUAAFCQkAIAAChISAEAABQkpAAAAAoSUgAAAAUJKQAAgIKEFAAAQEFCCgAAoCAhBQAAUJCQAgAAKEhIAQAAFCSkAAAAChJSAAAABQkpAACAgoQUAABAQUIKAACgICEFAABQkJACAAAoSEgBAAAUJKQAAAAKElIAAAAFCSkAAICChBQAAEBBQgoAAKAgIQUAAFCQkAIAAChISAEAABTUoUNq2rRpqaioaLHV1NSUj5dKpUybNi21tbXp2rVrhg8fnueff74dJwYAALYHHTqkkuTv/u7vUl9fX96effbZ8rFLL700l19+ea666qo8+eSTqampyYgRI7Jy5cp2nBgAANjWdfiQ6ty5c2pqaspbnz59kvzv2agrr7wy559/fsaOHZtBgwZl1qxZ+eMf/5jZs2e389QAAMC2rMOH1EsvvZTa2toMGDAgn/3sZ/PKK68kSZYuXZqGhoaMHDmyvLaysjLDhg3LY4899mcfs7m5OU1NTS02AACATdWhQ+rAAw/MjTfemHvvvTfXXnttGhoacvDBB+ett95KQ0NDkqS6urrFfaqrq8vHNmb69Ompqqoqb3V1dZvtNQAAANueDh1SY8aMyYknnpjBgwfnyCOPzK9+9askyaxZs8prKioqWtynVCqtt+9PTZ06NY2NjeVt2bJlbT88AACwzerQIfWnunfvnsGDB+ell14qX73vT88+LV++fL2zVH+qsrIyPXv2bLEBAABsqq0qpJqbm7N48eL069cvAwYMSE1NTebNm1c+vmbNmixYsCAHH3xwO04JAABs6zq39wB/zpQpU3Lsscdmjz32yPLly3PhhRemqakpp512WioqKjJp0qRcfPHFGThwYAYOHJiLL7443bp1yymnnNLeowMAANuwDh1Sr7/+ev7pn/4pf/jDH9KnT58cdNBBeeKJJ9K/f/8kyXnnnZfVq1fnrLPOyooVK3LggQdm7ty56dGjRztPDrBxr313cHuPAFulPb797F9eBLCFdOiQuu222/7s8YqKikybNi3Tpk3bMgMBAABkK/uOFAAAQEcgpAAAAAoSUgAAAAUJKQAAgIKEFAAAQEFCCgAAoCAhBQAAUJCQAgAAKEhIAQAAFCSkAAAAChJSAAAABQkpAACAgoQUAABAQUIKAACgICEFAABQkJACAAAoSEgBAAAUJKQAAAAKElIAAAAFCSkAAICChBQAAEBBQgoAAKAgIQUAAFCQkAIAAChISAEAABQkpAAAAAoSUgAAAAUJKQAAgIKEFAAAQEFCCgAAoCAhBQAAUJCQAgAAKEhIAQAAFCSkAAAAChJSAAAABQkpAACAgoQUAABAQUIKAACgICEFAABQkJACAAAoSEgBAAAUJKQAAAAKElIAAAAFCSkAAICChBQAAEBBQgoAAKAgIQUAAFCQkAIAAChISAEAABQkpAAAAAoSUgAAAAUJKQAAgIKEFAAAQEFCCgAAoCAhBQAAUJCQAgAAKGibCamrr746AwYMyE477ZQhQ4bk4Ycfbu+RAACAbdQ2EVK33357Jk2alPPPPz9PP/10/vEf/zFjxozJa6+91t6jAQAA26BtIqQuv/zyjB8/Pl/84hezzz775Morr0xdXV1mzJjR3qMBAADboK0+pNasWZOFCxdm5MiRLfaPHDkyjz32WDtNBQAAbMs6t/cAf60//OEPWbt2baqrq1vsr66uTkNDwwbv09zcnObm5vLtxsbGJElTU9PmG/T/t7Z59WZ/DtjWbIn/b25JK99b294jwFZpW3ov+GD1B+09AmyVtsT7wIfPUSqV/uy6rT6kPlRRUdHidqlUWm/fh6ZPn57vfOc76+2vq6vbLLMBf52qH/5Le48AdATTq9p7AqCdVX1ty70PrFy5MlVVG3++rT6kevfunU6dOq139mn58uXrnaX60NSpUzN58uTy7XXr1uXtt9/ObrvtttH4YtvW1NSUurq6LFu2LD179mzvcYB24H0ASLwX8L8nZFauXJna2to/u26rD6kuXbpkyJAhmTdvXk444YTy/nnz5uX444/f4H0qKytTWVnZYt8uu+yyOcdkK9GzZ09vmrCd8z4AJN4Ltnd/7kzUh7b6kEqSyZMn5/Of/3z233//DB06NP/+7/+e1157Lf/yLz4OBAAAtL1tIqROPvnkvPXWW/nud7+b+vr6DBo0KHfffXf69+/f3qMBAADboG0ipJLkrLPOyllnndXeY7CVqqyszAUXXLDeRz6B7Yf3ASDxXsCmqyj9pev6AQAA0MJW/4O8AAAAW5qQAgAAKEhIAQAAFCSkAAAAChJSbPcaGhoyceLEfOQjH0llZWXq6upy7LHH5v7772/v0YAtZNmyZRk/fnxqa2vTpUuX9O/fP1/+8pfz1ltvtfdowBayfPnynHnmmdljjz1SWVmZmpqajBo1Ko8//nh7j0YHtc1c/hxa49VXX80hhxySXXbZJZdeemk+9rGP5f3338+9996bCRMm5IUXXmjvEYHN7JVXXsnQoUOz11575dZbb82AAQPy/PPP56tf/WruueeePPHEE+nVq1d7jwlsZieeeGLef//9zJo1Kx/5yEfy+9//Pvfff3/efvvt9h6NDsrlz9muHXXUUXnmmWeyZMmSdO/evcWxd955J7vsskv7DAZsMWPGjMlzzz2XF198MV27di3vb2hoyEc/+tF84QtfyIwZM9pxQmBze+edd7Lrrrtm/vz5GTZsWHuPw1bCR/vYbr399tuZM2dOJkyYsF5EJRFRsB14++23c++99+ass85qEVFJUlNTk1NPPTW33357/JsjbNt23nnn7LzzzrnjjjvS3Nzc3uOwlRBSbLdefvnllEql7L333u09CtBOXnrppZRKpeyzzz4bPL7PPvtkxYoVefPNN7fwZMCW1Llz58ycOTOzZs3KLrvskkMOOSTf+MY38swzz7T3aHRgQort1of/wlxRUdHOkwAd1YfvE126dGnnSYDN7cQTT8wbb7yRO++8M6NGjcr8+fPz93//95k5c2Z7j0YHJaTYbg0cODAVFRVZvHhxe48CtJM999wzFRUV+e1vf7vB4y+88EL69Onjo76wndhpp50yYsSIfPvb385jjz2WcePG5YILLmjvseighBTbrV69emXUqFH5t3/7t7z77rvrHX/nnXe2/FDAFrXbbrtlxIgRufrqq7N69eoWxxoaGnLLLbdk3Lhx7TMc0O723XffDf43AiRCiu3c1VdfnbVr1+Yf/uEf8tOf/jQvvfRSFi9enB/84AcZOnRoe48HbAFXXXVVmpubM2rUqDz00ENZtmxZ5syZkxEjRmSvvfbKt7/97fYeEdjM3nrrrRx++OG5+eab88wzz2Tp0qX5yU9+kksvvTTHH398e49HB+Xy52z36uvrc9FFF+WXv/xl6uvr06dPnwwZMiRf+cpXMnz48PYeD9gCXn311UybNi1z5szJ8uXLUyqVMnbs2Nx0003p1q1be48HbGbNzc2ZNm1a5s6dm9/97nd5//33U1dXl8985jP5xje+sd5VPSERUgCwngsuuCCXX3555s6d6+w0ABskpABgA2644YY0NjbmnHPOyQ47+CQ8AC0JKQAAgIL8ExsAAEBBQgoAAKAgIQUAAFCQkAIAAChISAEAABQkpADYZk2bNi0f//jHN8tjz58/PxUVFXnnnXfa7DFfffXVVFRUZNGiRW32mABsHkIKgA5h3LhxqaioWG8bPXp0e48GAOvp3N4DAMCHRo8enRtuuKHFvsrKynaaZuPef//99h4BgHbmjBQAHUZlZWVqampabLvuumuSpKKiIj/60Y9yzDHHpFu3btlnn33y+OOP5+WXX87w4cPTvXv3DB06NL/73e/We9wf/ehHqaurS7du3fKZz3ymxcfxnnzyyYwYMSK9e/dOVVVVhg0blt/85jct7l9RUZFrrrkmxx9/fLp3754LL7xwvedYvXp1jj766Bx00EF5++23kyQ33HBD9tlnn+y0007Ze++9c/XVV7e4z69//et84hOfyE477ZT9998/Tz/99F/7VwjAFiKkANhqfO9738sXvvCFLFq0KHvvvXdOOeWUnHnmmZk6dWqeeuqpJMnZZ5/d4j4vv/xyfvzjH+euu+7KnDlzsmjRokyYMKF8fOXKlTnttNPy8MMP54knnsjAgQNz1FFHZeXKlS0e54ILLsjxxx+fZ599NmeccUaLY42NjRk5cmTWrFmT+++/P7169cq1116b888/PxdddFEWL16ciy++ON/61rcya9asJMm7776bY445Jn/7t3+bhQsXZtq0aZkyZcrm+GsDYHMoAUAHcNppp5U6depU6t69e4vtu9/9bqlUKpWSlL75zW+W1z/++OOlJKXrrruuvO/WW28t7bTTTuXbF1xwQalTp06lZcuWlffdc889pR122KFUX1+/wTk++OCDUo8ePUp33XVXeV+S0qRJk1qse/DBB0tJSi+88EJpv/32K40dO7bU3NxcPl5XV1eaPXt2i/t873vfKw0dOrRUKpVKP/rRj0q9evUqvfvuu+XjM2bMKCUpPf3003/x7wuA9uU7UgB0GIcddlhmzJjRYl+vXr3Kf/7Yxz5W/nN1dXWSZPDgwS32vffee2lqakrPnj2TJHvssUd233338pqhQ4dm3bp1WbJkSWpqarJ8+fJ8+9vfzgMPPJDf//73Wbt2bf74xz/mtddeazHH/vvvv8GZjzzyyBxwwAH58Y9/nE6dOiVJ3nzzzSxbtizjx4/Pl770pfLaDz74IFVVVUmSxYsXZ7/99ku3bt1azAbA1kFIAdBhdO/ePXvuuedGj++4447lP1dUVGx037p16zb6GB+u+fB/x40blzfffDNXXnll+vfvn8rKygwdOjRr1qxZb7YNOfroo/PTn/40v/3tb8tR9+HzX3vttTnwwANbrP8wtkql0kZnBKDjE1IAbNNee+21vPHGG6mtrU2SPP7449lhhx2y1157JUkefvjhXH311TnqqKOSJMuWLcsf/vCHTX78Sy65JDvvvHOOOOKIzJ8/P/vuu2+qq6vzN3/zN3nllVdy6qmnbvB+++67b2666aasXr06Xbt2TZI88cQTf81LBWALElIAdBjNzc1paGhosa9z587p3bt3qx9zp512ymmnnZbLLrssTU1NOeecc3LSSSelpqYmSbLnnnvmpptuyv7775+mpqZ89atfLYfNprrsssuydu3aHH744Zk/f3723nvvTJs2Leecc0569uyZMWPGpLm5OU899VRWrFiRyZMn55RTTsn555+f8ePH55vf/GZeffXVXHbZZa1+nQBsWa7aB0CHMWfOnPTr16/F9slPfvKvesw999wzY8eOzVFHHZWRI0dm0KBBLS5Dfv3112fFihX5xCc+kc9//vM555xz0rdv38LPc8UVV+Skk07K4YcfnhdffDFf/OIX8x//8R+ZOXNmBg8enGHDhmXmzJkZMGBAkmTnnXfOXXfdld/+9rf5xCc+kfPPPz/f//73/6rXCsCWU1HyIW0AAIBCnJECAAAoSEgBAAAUJKQAAAAKElIAAAAFCSkAAICChBQAAEBBQgoAAKAgIQUAAFCQkAIAAChISAEAABQkpAAAAAoSUgAAAAX9f3jANDzGIxRvAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 1000x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "prdata=df.groupby('Embarked').agg({'Survived':'count'}).reset_index()\n",
    "fig, (ax1) = plt.subplots(1,1,figsize=(10, 5))\n",
    "sns.barplot(x='Embarked', y='Survived', data = prdata, ax=ax1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "19ec25c3",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:47.318230Z",
     "iopub.status.busy": "2023-08-08T14:50:47.317781Z",
     "iopub.status.idle": "2023-08-08T14:50:48.450174Z",
     "shell.execute_reply": "2023-08-08T14:50:48.448877Z"
    },
    "papermill": {
     "duration": 1.158136,
     "end_time": "2023-08-08T14:50:48.452905",
     "exception": false,
     "start_time": "2023-08-08T14:50:47.294769",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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3nrrrUUVAwAAAAAAABRq8zv06urq8l89e/aMv/zlL/HUU0/l73/66afjL3/5S9TV1XVIoQAAAAAAALAiavM79K688sr8f59yyimx7777xiWXXBKVlZUREbF48eI46qijomfPnuWvEgAAAAAAAFZQRV1D74orrogf//jH+WFeRERlZWWccMIJccUVV5StOAAAAAAAAFjRFTXQ+/TTT+Oll15a5vaXXnoplixZUnJRAAAAAAAAwGfa/JGbjR166KFx2GGHxT//+c/YeuutIyLi8ccfjzPOOCMOPfTQshYIAAAAAAAAK7KiBnpnn3121NfXx7nnnhuzZs2KiIh+/frFySefHCeeeGJZCwQAAAAAAIAVWVEDvYqKijj55JPj5JNPjnnz5kVERM+ePctaGAAAAAAAAFDkNfQiPruO3gMPPBA33HBD5HK5iIh4++23Y8GCBWUrDgAAAAAAAFZ0Rb1D7/XXX49ddtkl3njjjVi4cGHsvPPO0aNHjzjzzDPj448/jksuuaTcdQIAAAAAAMAKqah36I0aNSqGDBkS7733XtTW1uZv32uvveIvf/lL2YoDAAAAAACAFV1R79B75JFH4tFHH40uXboU3D5w4MB46623ylIYAAAAAAAAUOQ79JYsWRKLFy9e5vY333wzevToUXJRAAAAAAAAwGeKGujtvPPOcd555+W/z+VysWDBgjjttNPiW9/6VrlqAwAAAAAAgBVeUR+5ee6558aOO+4YG2ywQXz88cdxwAEHxCuvvBKrrLJK3HDDDeWuEQAAAAAAAFZYRQ30+vfvH9OmTYsbbrghnnnmmViyZEkcfvjhceCBB0ZtbW25awQAAAAAAIAVVlEDvQ8//DC6desWhx12WBx22GHlrgkAAAAAAAD4f4q6hl6fPn3ioIMOivvuuy+WLFlS7poAAAAAAACA/6eogd4111wTCxcujL322iv69+8fo0aNiieffLLctQEAAAAAAMAKr6iB3siRI+OPf/xjvPPOOzFhwoR46aWXYtttt40vf/nL8Ytf/KLcNQIAAAAAAMAKq6iBXoMePXrEoYceGvfff3/8/e9/j+7du8fPf/7zctUGAAAAAAAAK7ySBnoff/xx3HTTTbHnnnvG5ptvHu+++278+Mc/LldtAAAAAAAAsMKrKuaH7r///rjuuuvi9ttvj8rKythnn33ivvvui6FDh5a7PgAAAAAAAFihFTXQ23PPPWPEiBFx9dVXx4gRI6K6urrcdQEAAAAAAABR5EBv9uzZ0bNnz3LXAgAAAAAAACylzQO9efPmFQzx5s2b1+y2hn0AAAAAAABQHm0e6K288soxa9as6NOnT3zpS1+KXC63zDYppcjlcrF48eKyFgkAAAAAAAArqjYP9P7nf/4nevXqlf/vpgZ6AAAAAAAAQHm1eaA3dOjQ/H/vsMMOHVELAAAAAAAAsJSKYn5orbXWip/97Gcxffr0ctcDAAAAAAAANFLUQO+YY46Je++9N9Zff/3YYost4rzzzotZs2aVuzYAAAAAAABY4RU10DvhhBPiySefjH/84x+x6667xsUXXxwDBgyIYcOGxTXXXFPuGgEAAAAAAGCFVdRAr8GXv/zl+PnPfx7Tp0+Phx9+OP73f/83Dj300HLVBgAAAAAAACu8qlIDnnjiibj++uvjxhtvjLlz58Y+++xTjroAAAAAAACAKHKg9/LLL8d1110X119/ffzrX/+KHXfcMc4444wYOXJk9OjRo9w1AgAAAAAAwAqrqIHeeuutF0OGDImjjz469t9//6ivry93XQAAAAAAAEAUMdBbvHhxXHLJJbHPPvtEr169OqImAAAAAAAA4P+paO8PVFZWxnHHHRdz587tiHoAAAAAAACARto90IuI2HjjjeO1114rdy0AAAAAAADAUooa6J1++unx4x//OO66666YNWtWzJs3r+ALAAAAAAAAKI92X0MvImKXXXaJiIjdd989crlc/vaUUuRyuVi8eHF5qgMAAAAAAIAVXFEDvQcffLDcdQAAAAAAAABNKGqgN3To0HLXAQAAAAAAADShqIHeQw891OL922+/fVHFAAAAAAAAAIWKGujtsMMOy9zW+Fp6rqEHAAAAAAAA5VFRzA+99957BV9z5syJe++9N7bccsu4//77y10jAAAAAAAArLCKeodeXV3dMrftvPPOUVNTE8cff3w8/fTTJRcGAAAAAAAAFPkOveasuuqqMX369HJGAgAAAAAAwAqtqHfoPfvsswXfp5Ri1qxZccYZZ8Smm25alsIAAAAAAACAIgd6X/nKVyKXy0VKqeD2rbfeOq644oqyFAYAAAAAAAAUOdCbMWNGwfcVFRWx6qqrRteuXctSFAAAAAAAAPCZdl1D729/+1v8+c9/joEDB+a/Jk+eHNtvv30MGDAgfvjDH8bChQs7qlYAAAAAAABY4bRroDdu3LiC6+c999xzcfjhh8dOO+0Up556atx5550xYcKEshcJAAAAAAAAK6p2DfSmTZsW3/jGN/LfT5w4Mbbaaqu47LLL4oQTTogLLrggbrrpprIXCQAAAAAAACuqdg303nvvvejbt2/++8mTJ8cuu+yS/37LLbeMmTNnlq86AAAAAAAAWMG1a6DXt2/fmDFjRkRELFq0KJ555pnYZptt8vfPnz8/qqury1shAAAAAAAArMDaNdDbZZdd4tRTT42HH344xowZE926dYuvf/3r+fufffbZWHvttcteJAAAAAAAAKyoqtqz8S9/+csYOXJkDB06NFZaaaW4+uqro0uXLvn7r7jiihg2bFjZiwQAAAAAAIAVVbsGequuumo8/PDDMXfu3FhppZWisrKy4P4//vGPsdJKK5W1QAAAAAAAAFiRtWug16Curq7J23v16lVSMQAAAAAAAEChdl1DDwAAAAAAAPh8GegBAAAAAABAhhnoAQAAAAAAQIYZ6AEAAAAAAECGGegBAAAAAABAhhnoAQAAAAAAQIYZ6AEAAAAAAECGGegBAAAAAABAhhnoAQAAAAAAQIYZ6AEAAAAAAECGGegBAAAAAABAhhnoAQAAAAAAQIYZ6AEAAAAAAECGGegBAAAAAABAhi1XA70JEyZELpeL0aNH529LKcW4ceOif//+UVtbGzvssEO88MILnVckAAAAAAAAlNFyM9B78skn49JLL41NNtmk4PYzzzwzzjnnnLjwwgvjySefjPr6+th5551j/vz5nVQpAAAAAAAAlM9yMdBbsGBBHHjggXHZZZfFyiuvnL89pRTnnXdejB07NkaOHBkbbbRRXH311fHhhx/G9ddf34kVAwAAAAAAQHksFwO9o48+OkaMGBE77bRTwe0zZsyI2bNnx7Bhw/K31dTUxNChQ2PKlCnN5i1cuDDmzZtX8AUAAAAAAABZVNXZBbRm4sSJ8cwzz8STTz65zH2zZ8+OiIi+ffsW3N63b994/fXXm82cMGFC/PznPy9voQAAAAAAANABMv0OvZkzZ8aoUaPi2muvja5duza7XS6XK/g+pbTMbY2NGTMm5s6dm/+aOXNm2WoGAAAAAACAcsr0O/SefvrpmDNnTmyxxRb52xYvXhwPPfRQXHjhhTF9+vSI+Oydev36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AAAAAAAAKMkXYqC33377xbvvvhu/+MUvYtasWbHRRhvFPffcEwMHDiz7v1VTUxOnnXbaMh/ZmbXMjspdnmrtqFy1Ll+5alVrR+WqdfnKVataOypXrWrtqFy1Ll+5alVrR+WqdfnKVataOypXrctXrlrV2lG5al2+cjsiM5dSSmVLAwAAAAAAAMpqub+GHgAAAAAAAHyRGegBAAAAAABAhhnoAQAAAAAAQIYZ6AEAAAAAAECGfSEGehdddFEMGjQounbtGltssUU8/PDDzW47a9asOOCAA2LdddeNioqKGD169DLb7LDDDpHL5Zb5GjFiRJOZjzzySGy33XbRu3fvqK2tjfXWWy/OPffcgm1uvfXWGDJkSNTW1kZFRUVUVFTEoEGDWqz1kEMOabKODTfcML/NVVdd1eQ2H3/8cbO51113XWy66abRrVu36NevXxx66KHx7rvvllRrRMRvf/vbWH/99aO2tjbWXXfduOaaawruv+yyy+LrX/96rLzyyrHyyivHTjvtFE888UTBNg899FDstttu0b9//8jlcnH77be3+G9GREyePDm22GKL6Nq1a6y11lpxySWXFNw/YcKE2HLLLaNHjx7Rp0+f2HPPPWP69Okl51588cWxySabRM+ePaNnz56xzTbbxJ///OeSMseNG7fMY1lfX19SZoO33norDjrooOjdu3d069YtvvKVr8TTTz9dUvaaa67Z5P539NFHF5356aefxk9/+tMYNGhQ1NbWxlprrRW/+MUvYsmSJSXVOn/+/Bg9enQMHDgwamtrY9ttt40nn3yyxcwLLrgg6urq8r/XkUceWXB/SinGjRsX/fv3j9ra2thhhx3ihRdeaDHzoYceiiFDhkR1dXXkcrlYY4014rbbbsvff+utt8bw4cNjlVVWiVwuF9OmTWsxL6L1PvDJJ5/EKaecEhtvvHF07949+vfvH9/97nfj7bffbjX7v//7v6NHjx75zFNPPbXg/nHjxsV6660X3bt3zx/Xf/vb30pag8aOOOKIyOVycd5555VcazG5Y8aMaXFtm+rTW2+9dVlqfemll2L33XePurq66NGjR2y99dbxxhtvNJvZ2ro29Xvkcrk466yzSqp1wYIFccwxx8Tqq68etbW1sf7668fFF19c8hoUk9va4/XOO+/EIYccEv37949u3brFLrvsEq+88kqLmRMmTIh11lknKisrI5fLRY8ePeLCCy8s2KaYXtCW3Pb2g4bnu9ra2qiqqoqKiopYe+218/tBsb2gtVqLzd1nn31aPYdpb49pbQ2W1tZe0JbHq739oK21trcXtKXW9uZOmDAhBg0a1OLjVUyPaUutxfSCjshtyxoU02MuvvjiGDhwYL7Wbt26xc9+9rP8/cX0l9YyiznXiIj44Q9/GF27do1cLheVlZWx7rrr5s99SznX6Ijc7373uy0+VsWcv7RW69La2l9ae7yKyWzu3PBPf/pTfptiz2Hasgbt7VttWYP2ZrZlDYrpW63VWux5UWvrWkxua8dBMT0rIuKWW26JDTbYIGpqamKDDTaIgw46KHK5XMFrK8X0rsbZVVVVkcvlYtdddy0ps7n94JhjjslvU2xPbK7WUvph47Xt06fPMutabO9qLbex9vz91dLjVWyPaUut7e0HrdVa7N9JrdVabD9oqdZijtumXmdaeeWV8/cXe7zuu+++y+Q2fv2q2GOrtdxijoPW1qCx9hwDrdVazHHQWq3F7q+t1Vrs/tpabrHPNZdeemn+9biG16Ubv4ZZ7H7bEbnnnntui8+3xR4LLdVa7HPNW2+9Fdtvv32+v3Tt2jXOPvvs/P3FPs+0tq6NtecYay23vcdDc69jf/Ob38xvU8pzTMO5UXOvP5RNWs5NnDgxVVdXp8suuyy9+OKLadSoUal79+7p9ddfb3L7GTNmpOOOOy5dffXV6Stf+UoaNWrUMtv8/ve/T9XV1enss89OkydPTgcddFCKiHT22Wc3mfnMM8+k66+/Pj3//PNpxowZ6Q9/+EPq1q1b+t3vfpff5sEHH0wnnHBCqqqqSqeffno69dRTUy6XS127dm221vfffz/NmjUr/zVz5szUq1evdNppp+W3OfLII/O1TZ48OX3/+99P3bp1azbz4YcfThUVFen8889Pr732Wnr44YfThhtumPbcc8+Sar3oootSjx490sSJE9Orr76abrjhhrTSSiulO+64I7/NAQcckH7729+mqVOnppdeeikdeuihqa6uLr355pv5be655540duzYdMstt6SISLfddluT/16D1157LXXr1i2NGjUqvfjii+myyy5L1dXV6eabb85vM3z48HTllVem559/Pk2bNi2NGDEiDRgwIC1YsKCk3DvuuCPdfffdafr06Wn69OnpJz/5Saqurk7PP/980ZmnnXZa2nDDDQse9zlz5pRUZ0op/ec//0kDBw5MhxxySPrb3/6WZsyYkR544IH0z3/+s6TsOXPmFNQ6adKkFBHpwQcfLDrzl7/8Zerdu3e666670owZM9If//jHtNJKK6XzzjuvpFr33XfftMEGG6TJkyenV155JZ122mmpZ8+eBfvf0pk1NTVpyy23TOeff36KiFRZWVmQecYZZ6QePXqkW265JT333HNpv/32S/369Uvz5s1rttZf//rXKZfLpQMOOCBFRDrooINSVVVVevzxx1NKKV1zzTXp5z//ebrssstSRKSpU6c2m9XgyiuvTLW1tWnUqFHp97//fYqIdMUVV+Tvf//999NOO+2UbrzxxvSPf/wjPfbYY2mrrbZKW2yxRYu5U6ZMSRUVFWmHHXbIr0FFRUW+1pRSuu6669KkSZPSq6++mp5//vl0+OGHp549e7a437a2Bg1uu+22tOmmm6b+/func889t+Rai8k9/vjjU5cuXQrWddasWfn7v/e976Vddtml4Dh49913S671n//8Z+rVq1c66aST0jPPPJNeffXVdNddd6V33nmn2dzW1rVxjbNmzUpXXHFFyuVy6dVXXy2p1u9///tp7bXXTg8++GCaMWNG+t3vfpcqKyvT7bff/rnntvR4LVmyJG299dbp61//enriiSfSP/7xj/TDH/6w1eeDrbbaKuVyuTRq1Kh02223pXXXXXeZPldML2hLbnv7wfDhw9PYsWNTRUVFOvbYY9PQoUPTl770pVRZWZkef/zxontBa7UWm7vRRhulrl27pr/+9a/pgQceSDvttFNabbXVCh6P9vaY1tagsfb0grY8Xu3tB22ptZhe0JZa25s7fPjwdPjhh6fu3bsXPFaN+0cxPaYttRbTCzoit7U1KLbHnHnmmamioiKdcMIJ6Z577kk77LBDioh0/fXXp5SK6y+tZRZzrtHQt7/3ve+le+65J51wwgkpl8ulqqqq9Pzzz5d8rlHu3FGjRqXa2tr0yCOPpEceeSQdd9xxBefpxZy/tFZrY+3pL609XsVkNpwbXnvttc2uQSnnMC2tQTF9q7U1KCazLWtQTN9qrdZSzotaWtdicls6DortWVOmTEmVlZVp/Pjx6aWXXkpHHXVUioi0zjrrFLy2Ukzvasg+6qij0mqrrZbq6+tTLpfLPycWk3nllVemnj17plmzZqU///nPaY011kgbbLBBQa3F9sTmai2lHzas7U033ZRWXnnlFBFpv/32y29TbO9qLbdBe//+aunxKrbHtFZrMf2gtVqL/TuptVqL7QfN1VrscXvYYYeliEhjxoxJDz30UBozZkzB+Waxx2sul0t9+vQpyPzzn/+c36bYY6u13GKOg9bWoEF7j4HWai3mOGit1mL319ZqLXZ/bSm32H323nvvTRGRNt9883TjjTemk046KVVUVJT8elxH5P7nP/9JvXv3TtXV1emee+5JTzzxRLrpppvSY489lt+mmGOhtVqLea75z3/+k+8pP/rRj9IDDzyQDjvssIL9q5jjqy3r2qA9x1hbctt7PMyZMyfdeeedqaKiIo0ZMyZdfvnlBa9HlfIc03BuNH78+CZf4yyn5X6g99WvfjUdeeSRBbett9566dRTT231Z4cOHdrkQG/pzHPPPTdVVFSkE088sc117bXXXumggw5qMXezzTZLvXv3blOtKX220+dyufSvf/0rf9taa62VunTpUrBdS7//WWedldZaa62C2y644IK0+uqrl1TrNttsk3784x8X3DZq1Ki03XbbNfv7fPrpp6lHjx7p6quvbvL+tgz0Tj755LTeeusV3HbEEUekrbfeutmfmTNnToqINHny5LLmppTSyiuvnH7/+98XnXnaaaelTTfdtMV/o5g6TznllPS1r32tzbntyW5s1KhRae21105LliwpOnPEiBHpsMMOK9hm5MiRyxxP7cn98MMPU2VlZbrrrrsKttl0003T2LFj25QZEWnYsGH5zCVLlqT6+vp0xhln5Lf5+OOPU11dXbrkkkuarXXfffdNu+yySz7ztttuS8OHD0/7779/wXYzZsxo10Cvrq6uoNbWjp0nnngiRUSzQ/qla23I/cpXvrJMrY3NnTs3RUR64IEH2pTb3Bq8+eababXVVkvPP/98GjhwYKtP9G2ttb25jde2qXX93ve+l/bYY48WM4qpdb/99mtxn28tt6V9q8Eee+yR/s//+T8l17rhhhumX/ziFwU/t/nmm6ef/vSnn3tuS4/X9OnTU0QUvOj66aefpl69eqXLLruszbU2PIc0rF05ekFTuY21px80zm3I3HLLLZvdD4rpBS3V2p7cpXtXW56f29tjmluDUntMU2tQaj9oqtZSe0FztRaT2/jxastjVUyPaarWcvSYcuW2tAbl6jEppVRVVZW23HLLsvWXxpmNFdtbGgwfPjz/P1E0pZj+Uq7cpftLSi2fp7e3t7RUa6n9JaVlH69Szl8aa7wG5TiHSWnZNShH30qpcA1K7VmNtbQfFNO3lq61HD0rpWXXtdSe1aDh9y9Hz5o/f34aPHhwGjJkSFp11VXzr62U0rt22mmnNHjw4DRp0qQ0dOjQNGDAgLT//vsXndmwBg21NuQ29TpQe3tic7U2pT39sHGtK6+8cvryl7/c7M+0p3e1lltM72ppDUrpMS3VWmyPac/j1Z5e0FKtxfaD5mot9rjdYIMNUo8ePQpua/g7sZTjde211y54/aq5vz3be2y1NbdBW46DltagQTHHQGu1FnMctKXWxtq6v7ZWa7H7a0u5xe6z6623Xlp55ZULbmucW+x+2xG5p5xySho8eHCT5xxLa8+x0FqtTWntueaUU05Jq6yySpPnHM3ltuX4amut7T3GilmD9p7LNbyOPWzYsLT//vuX7Ty2tTpLtVx/5OaiRYvi6aefjmHDhhXcPmzYsJgyZUrZMi+//PJYf/31W/14vgZTp06NKVOmxNChQ5vMTSnFX/7yl5g+fXp8/etfb3Otl19+eey0004xcODAfOa//vWv+PTTT2PgwIGx+uqrx6677hqbbbZZs5nbbrttvPnmm3HPPfdESineeeeduPnmmws+TrSYWhcuXBhdu3YtuK22tjaeeOKJ+OSTT5r8mQ8//DA++eST6NWrV5t+/6Y89thjyzz+w4cPj6eeeqrZf3fu3LkRES3+u+3NXbx4cUycODE++OCD2GabbUrKfOWVV6J///4xaNCg2H///eO1114ruc477rgjhgwZEt/+9rejT58+sdlmm8Vll13WbG57shssWrQorr322jjssMMil8sVnfm1r30t/vKXv8TLL78cERF///vf45FHHolvfetbRdf66aefxuLFi5vcRx955JE2Z2622Wb5zBkzZsTs2bMLtqmpqYmhQ4e2eEw3V2uxPavBggUL8n0gIlrcbyI+Ow5yuVx86UtfaletLfWXRYsW5d8Ov+mmm7Yrt/EaLFmyJA4++OA46aSTCj5iuCVtqbWY3Ij/v7YREb/85S9j6tSpBff/9a9/jT59+sSXv/zl+MEPfhBz5swpqdYlS5bE3XffHV/+8pdj+PDh0adPn9hqq61a/Qji9uxb77zzTtx9991x+OGHl1RrxGfH7B133BFvvfVWpJTiwQcfjJdffjmGDx/+uedGNP94LVy4MCKioA9UVlZGly5dmu0DTdXa8Bzy4osvRkSUrRcsnVusxrkNmTvssEOztRTTC9pSa1tyIwp717777hsRzT8/F9NjmlqDcvSY5taglH6wdK3l6gVL11psbsT/f7w23njjiPislzSl2B7T1LqWo8eUKzei+TUoR49pOJ9MKcWbb75Zlv6ydGaxll7TxYsXR58+fWLRokXNnvsW01/KlRvx/x+r1VZbLTbffPNYsGBBk5nFnr80VWup/aWpx6vU85fVV189RowYEWecccYyf6uUeg6z9BqUo28tvQbl6FktrUGDYvpWU49XOXpWU/tWqT1r6eOgHD3r6KOPjhEjRsQBBxwQ8+bNy29TSu96//33Y8SIEbHTTjtFRMTAgQNjypQpRWc2rEF9fX28/fbbcd5558WCBQta3L4tWqq1Ke3phw3rutNOO0WvXr1i1qxZTW7f3t7VUm6xvau1NSi2xzRXayk9pq2PV3t7QUvrWmw/aK7WYo/bN998MxYuXFjwOtOQIUNKOrYee+yxWHvttQtev3rnnXfioYceanHdWtPe3LYeBy2tQUTxx0Bbam3vcdBarY21Z39trdZi99eWcovdZ//5z3/GxhtvXPAaZs+ePfNrUOx+2xG5d9xxRwwaNCjmzp2b/9022WSTZV43aq/Wam1Ka881d9xxR3z88ccxa9asgteGm3vdqK3HV1tqLeYYa+8atLd/N34de5dddolHH320w1+LK5sOGxV+Dt56660UEenRRx8tuP30009v8f9iatDU/5m1dObf/va3FBHpRz/6UauZq622WurSpUuqqKhY5v9qaMjt2rVrqqqqSjU1Nenyyy9vc61vv/12qqysTDfeeOMymT/72c/StGnT0kMPPZT23nvvVFVVlQYNGtRsVsNHGFZVVaWISLvvvntatGhRSbWOGTMm1dfXp6eeeiotWbIkPfnkk6lPnz4pItLbb7/d5M8cddRRae21104fffRRk/dHG95lNHjw4HT66acX3Pboo482++8uWbIk7bbbbq2+W62tuc8++2zq3r17qqysTHV1denuu+8uKfOee+5JN998c3r22Wfz/zdW375907///e+S6qypqUk1NTVpzJgx6ZlnnkmXXHJJ6tq1a7PvjmxPdoMbb7wxVVZWprfeequkzCVLluQ/5rWqqirlcrk0fvz4ZjPbmrvNNtukoUOHprfeeit9+umn6Q9/+EPK5XLN7tNLZ0ZEmjBhQj6zIX/p3/cHP/hBGjZsWLO1VldXp+uuuy6fedttt6XrrrtumXfatuf/3HnsscfSH/7wh3wfiIjUpUuX9PLLLze5/UcffZS22GKLdOCBB7aY27jWhnobPtKwsTvvvDN179495XK51L9///TEE0+0ObepNRg/fnzaeeed8+/0bMv/udOWWovJbby2EZG23nrrVFtbm1/biRMnprvuuis999xz6Y477kibbrpp2nDDDdPHH39cdK2zZs1KEZG6deuWzjnnnDR16tQ0YcKElMvl0l//+tc25ba0b6WU0q9+9au08sorN9t/21prSiktXLgwffe7300RkaqqqlKXLl3SNddc0ym5LT1eixYtSgMHDkzf/va303/+85+0cOHC/DHd1mO24Tlk3XXXzddajl7QVG5j7ekHDbmNn++a2w+K6QWt1dqe3MaP1+TJk1O/fv1SRUXFMr2r2B7T3BqU2mOaW4NS+kFTtZajFzRVa7G5DY/X1KlT07bbbpt69+5d0A8bK6bHNLeupfaYcua2tAal9JgJEyYUnE+edNJJqUuXLiX1l+YyGyumtzQ+9+3WrVuqqqpqcvv29pdy5z722GNp/Pjxqba2NlVWVqbq6uplzo2K7S0t1Vpsf2np8Srl/OWPf/xjqq2tTblcLkVEuvTSS/PblNKzmluDUvpWc2tQas9qaQ0aa0/faunxKqVntbRvFduzmjsOSj0vuuGGG9JGG22UPvroo3TdddflP+Y4peLPjSorK9Pqq6+efwyGDh2adtlll5L64WOPPZaOOuqotM4666RJkyalvffeO//xpktrT09sqdaltacfHnPMMfl1TSml9ddfP1VUVBRsV0zvai23mD7T2hoU22NaqrXYftCex6s9vaC1dS3muG2p1mKP28rKyjRq1KiC15nq6upSdXV1SecaJ510UsHrV+uvv36KiGVev2rv+UZbctt7HLS0BikV//zdWq3FHAet1dpYe/bX1mot9vmrpdxi99mISNXV1QWvYVZXV6fKysqUUvHPMx2RW1NTk7p06ZJ22223NHHixHTiiSemioqKVF1dvczfSe05FlqrdWltea6pqanJzwEavzZ85JFHFvTD9h5fbam1mGOsvWvQ3r9BG7+Ofd1116Xq6uqS//5u0NxrMOXyhRjoTZkypeD2X/7yl2nddddt9edbGug1ZP7whz9MG220UZsyX3vttfTss8+mSy+9NPXq1avgugcNuTfddFOaOnVqOvvss1NdXV06/PDD21Tr+PHjU+/evdPChQubrTWllBYvXpzq6+vTl770pSZzXnjhhdSvX7905plnpr///e/p3nvvTRtvvHHBRxwWU+uHH36YDj300FRVVZUqKytT//7908knn5wiosnPmW04yP7+9783+zu3daC39LDnkUceSRFRcJ2rBkcddVQaOHBgmjlzZllyFy5cmF555ZX05JNPplNPPTWtssoq6YUXXihLrSmltGDBgtS3b9/061//uqTM6urqtM022xRsd+yxx7b48ZntrXfYsGFp1113bTavrZk33HBDWn311dMNN9yQnn322XTNNdekXr16pauuuqqk3H/+859p++23TxGfXQtvyy23TAceeGBaf/3125QZEWn8+PH5zOaGm9///vfT8OHDm621uro63xsa9vFrr7021dTUFGzXnif6pUVEWnPNNdOxxx67zH2LFi1Ke+yxR9pss83S3LlzW8xpXGtD7ujRo5epdcGCBemVV15Jjz32WDrssMPSmmuu2eLnS7e0Bk899VTq27dvwQlUW0+mW6q12NzGIiLdcsstadNNN21ybVP67H++qK6uTrfcckvRtTb04O985zsFP7fbbru1+Jb9tu5bKaW07rrrpmOOOab5X7aNtab02Uc5f/nLX0533HFH+vvf/55+85vfpJVWWilNmjTpc89trKnH66mnnkqbbrppvg8MHz48ffOb30zf/OY321Rrw3PI+eefn6+1HL2gqdzG2vtH8PXXX1/wfNfUflBsL2it1vbkNtaQu/766y9zfBXbY5pag3L0mNbWoEF7+kFTtZajFzRVa7G5DRoyX3/99Wb7YTE9prl1LbXHlDO3tTUotsdcc801BeeTPXr0KHiRrZj+0lxmY8X0lsbnvrvttluKiGXOfYvpL+XOTanwPP2UU05JlZWVBS9yFNtbmqu1lP7S3ONV6vlLQ61/+9vfUp8+fVLXrl2b/VulPT2ruTUopW81twal9qy2rkF7+lZLx1cpPaul46DYntXScVBsz/rNb36T+vTpk6ZNm5ZSSunaa69tcqDXnt71xhtv5P8HygZDhw5Nw4cPTzU1NUX3wzfeeKOg1sWLF6fu3bs3eZmLtvbE1mptrD19q6qqKvXs2TNfa0qffdzY0i9ctrd3tZZbTJ9pzxo0aEuPaa3WYvpBe2ttay9oy+PV3uO2LbWW+vdMSp/tQ3V1damysrIsf8s0aLiO+dKvXxVzvtFabimvQTT8fMMalOs1iOZqbayY1wsa17q0Yv+ub6rWcvxd31RuMftsRKTBgwcX3DZs2LCUy+VSSsX/Dd4RuU293nrMMcekbt26LfN3UnsHei3V2lhbn2uqq6tTLpcreMyOPfbYtM466xT0w/YeX63VWuwx1p41SKn9x0Pj17Gvvfba1KVLl5L//m7Q3Gtx5bJcf+TmKqusEpWVlTF79uyC2+fMmRN9+/YtOfPDDz+MiRMnxve///02ZQ4aNCg23njj+MEPfhDHH398jBs3bpncqqqq+MpXvhInnnhi7LPPPjFp0qRWc1NKccUVV8TBBx8cXbp0abLWBhUVFdG7d+9mP/JwwoQJsd1228VJJ50Um2yySQwfPjwuuuiiuOKKK/IfDVBMrbW1tXHFFVfEhx9+GP/617/ijTfeiDXXXDN69OgRq6yySsG2Z599dowfPz7uv//+2GSTTVr83VtTX1/f5ONfVVUVvXv3Lrj92GOPjTvuuCMefPDB/McSlprbpUuXWGeddWLIkCExYcKE2HTTTeP8888vudYG3bt3j4033jheeeWVkjL79esXG2ywQcF266+/frzxxhtN5ra33tdffz0eeOCB+P73v99sXlszTzrppDj11FNj//33j4033jgOPvjgOP7442PChAkl5a699toxefLkWLBgQcycOTP/cbCDBg1qc+bcuXPzmfX19RER7e4/zdVabM9qzjrrrLPMfvPJJ5/EvvvuGzNmzIhJkyZFz549W8xobg2WrrV79+6xzjrrxNZbbx2XX355VFVVxeWXX96u3IY1ePjhh2POnDkxYMCAqKqqiqqqqnj99dfjxBNPjDXXXLPoWovNXVpFRUVsueWWzR6T/fr1i4EDBzZ7f1tqXWWVVaKqqqpsx+zSj9fDDz8c06dPb/V4bUutH330UfzkJz+Jc845J3bbbbfYZJNN4phjjon99tsvzj777M89d2lLP15bbLFFTJs2Ld5///2YNWtW3HvvvfHuu+822wca19r4OWTx4sX5WkvtBc3lFqu+vj4uuuiigue7pWspthe0Vmt7cxs0zt1uu+2WOX6K6THNrUGpPaY9j1db+0FztZbaC5qrtdjciMLHasCAAU32w2J6THO1ltpjyp3b2hoU22P+/e9/F5xP1tfXR01NTUn9pbnMYjWsaeNz3x133DFqamoKzn2L7S/lzo0oPE8/44wzom/fvjF58uT8/cWevzRXayn9pbnHq9Tzl4Zav/rVr8buu+8e3bt3b/ZvlfacwzS3BqX0rebWoJSe1dY1aG/faq7WUntWc+taSs9q6Tgotmc988wzMWfOnNhiiy2iqqoqvvvd70ZKKS644IKoqqrK96f29K6nn346IiLGjh2b39cnT54c999/fyxcuLCozIbcxrV26dIlPvjgg/j73/8eVVVVsXjx4hbXr5haGzLb27fq6upi3rx5+VqrqqriH//4RyxevLig1vb2rtZy//rXv7a7z7R1DRprS49prdbevXu3ux+0p9b29ILWav3ggw/afdy2pdZS/p5p0L179+jbt2/U1taW9XWNefPmRdeuXVt8jFvT1txSX4NovAblfA2itTUo5vWCxrU2Vurf9Y1rLeff9UuvQTH7bGVl5TKXYejRo0dUVFTk/92I9u+3HZHb1OutDd+Xciy0VmuD9jzX9OvXL7p161bw+62//vrL/H7tPb5aq7XYY6yta9Dwb7TneHjxxRcLXsduWIOOfC2unJbrgV6XLl1iiy22iEmTJhXcPmnSpNh2221Lzrzpppti4cKFcdBBB7U7M6WU/6zg5mpNKcW///3vVnMnT54c//znP5f5DNjmMl999dVmB1YffvjhMjt+ZWVl/mdLrbW6ujpWX331qKysjIkTJ8auu+5a8O+dddZZ8d///d9x7733xpAhQ1rMaottttlmmcf//vvvjyFDhkR1dXW+9mOOOSZuvfXW+J//+Z8WnzTak9uUpR/3UjMXLlwYL730UvTr16+kzO222y6mT59esN3LL7+cv85UqfVeeeWV0adPn4JrMRab2dw+umTJkrLU2r179+jXr1+89957cd9998Uee+zR5sxp06blMwcNGhT19fUF2yxatCgmT57c4nHSXK3F9qzmzJgxo2C/aXiSf+WVV+KBBx5odojcWq3Tpk1rtdaWjoPmchvW4OCDD45nn302pk2blv/q379/nHTSSXHfffcVXWuxuU39btOmTWv2mHz33Xdj5syZzd7fllq7dOkSW265ZdmO2aUfr8svvzy22GKLFj8Dva21fvLJJ/HJJ5+U5ZgtR+7Smnu86urqYtVVV41XXnklnnrqqWb7QETE1ltvHeedd17Bc0jjdS22F7SWW4yUUtTU1MSTTz5Z8HzXOLeYXtCWWovJXfr5ec0112zx+Gr8c831mNbWoNheUMzj1Vo/aK3WYntBa7UWk9vUuVRzx1d7ekxrtRbbCzoitz1r0J4e01Q/fPfdd2OVVVYp67lGQ2axmnuOqauryx+P5TrXKEfu0lJKMX/+/GWup7z0NsWcvzTUWs5zmIbHq9znL126dGn2dyz2HKbxGpTzHKZhDYrNXFpLa1DquVFDreU8L2q8ruU8L2rqOGhvz3rzzTfjueeey++T22yzTfTq1SsOPPDAmDZtWqy11lrt7l3f+MY3Yvjw4bHtttvmc4cMGRL9+vWLXXbZpajMhtzGtU6dOjW6desWa6+9dkybNi3/Wkh7tFZrZWVlUX1r++23j6997WsFx3vPnj1j4MCBLdbaWu9qLfeQQw5pd59pyxosrS09prVaa2pq2t0P2lNre3pBa7UuXry43cdte2ot5Vxj4cKF8frrr8eaa65Z1nONe++9N3K5XKvn8i0pNre9z+GN16Ccz9+t1VrMc23jWhsr9bmrca3lfP5qbg3as8+uttpq8fLLLxfc9tRTT+WvDVfsftsRuU293jp9+vSSj4XWao1o/znydtttF7W1tQW/38svvxwVFRUtrltrx1drtRZ7jLVlDRq093iYOHFiwevY999/f2y33XYd+lpcWXXYe/8+JxMnTkzV1dXp8ssvTy+++GIaPXp06t69e/rXv/6VUkrp1FNPTQcffHDBz0ydOjVNnTo1bbHFFumAAw5IU6dOLfjYjYbMddZZJ33zm99sNfPCCy9Md9xxR3r55ZfTyy+/nK644orUs2fPNHbs2Pw248ePTz/5yU9SdXV1+uUvf5lOPvnklMvlUk1NTYu1ppTSQQcdlLbaaqsmf/999tknVVVVpTPOOCPdcsstaYMNNkgRkW6//fYmM6+88spUVVWVLrroovTqq6+mRx55JA0ZMiR99atfLanW6dOnpz/84Q/p5ZdfTn/729/Sfvvtl3r16pVmzJiR3+ZXv/pV6tKlS7r55pvTrFmz8l/z58/PbzN//vz84xMR+c+sff3115v8d1977bXUrVu3dPzxx6cXX3wxXX755am6ujrdfPPN+W1+9KMfpbq6uvTXv/614N/98MMP89sUkztmzJj00EMPpRkzZqRnn302/eQnP0kVFRXp/vvvLzrzxBNPTH/961/Ta6+9lh5//PG06667ph49ejS77m3JTCmlJ554IlVVVaXTTz89vfLKK+m6665L3bp1S9dee21Ja5DSZx9bMmDAgHTKKaekpRWT+b3vfS+tttpq6a677kozZsxIt956a1pllVXSySefXFLuvffem/785z+n1157Ld1///1p0003TV/96lfz149sKrO2tjYdeOCB6ZZbbkkRkSoqKtJZZ52V3x/POOOMVFdXl2699db03HPPpe985zupX79+ad68efmcgw8+OJ166qn57ydNmpQqKirScccdlyIijRgxIlVWVuaP2XfffTdNnTo13X333Ski0sSJE9PUqVMLPuZ06cxx48al2267Ld15551p4sSJ+Vr/8Ic/pNdffz198sknaffdd0+rr756mjZtWsFx0PhjfJfOffTRR/O13nrrrfnca665Jr3++utpwYIFacyYMemxxx5L//rXv9LTTz+dDj/88FRTU5Oef/75otdgaU29Fb+9tRabO2bMmPTb3/423XnnnSki0le/+tVUWVmZ/vSnP6X58+enE088MU2ZMiXNmDEjPfjgg2mbbbZJq622Wov7QFtqvfXWW1N1dXW69NJL0yuvvJJ+85vfpMrKyvTwww+XtK5z585N3bp1SxdffHGTa1JMrUOHDk0bbrhhevDBB9Nrr72WrrzyytS1a9d00UUXfe65LT1eKaV00003pQcffDC9+uqr6fbbb08DBw5MI0eObHEN9tprrxQR6Yc//GF66KGH0tixY1NlZWXBZ6gX0wvaktvefvCjH/0orbTSSqmioiKNHTu2IPfxxx8vuhe0VmuxuUOGDEndu3dP119/fZo0aVLab7/9UmVlZZo8eXJKKRXVY1pbg6a0pRe0tgbF9IO21FpML2jLvtXe3B/96EeppqYmnXnmmenxxx8veLz+9re/5X+mvT2mLbUW0ws6Ircta1BMjzn44INTRUVFOvnkk9Of/vSntOOOO6aISOeff35Kqbj+0lpmMecajz76aMrlcumII45IDzzwQBo9enTK5XL5c99SzjU6IvdrX/taOvvss9PkyZPTTTfdlDbZZJMUEemCCy4o+vyltVqb0pb+0trjVUzmuHHj0n777ZcmTpyY7r777rTHHnsU1FrKOUxra1BM32ptDYrJbG0NGrS3b7VWa7HnRa2tazG5LR0HKRXXsx599NFUWVmZzjjjjPTSSy+lM844I1VVVaXNNtus4HImxfSupbMHDRqUcrlc/jmxmMxx48ale++9N7366qtp6tSp6dBDD00Rkfbbb7/8NsX2xOZqLaUfLr22jWstpXe1lNuUtv791dwalNJjWqu1mH7Q2r6VUnF/J7VWa7H9oKVaizluv/Od7+T71sSJE9N6661XcImbYo/XXC6XfvjDH6ZJkyalo446KuVyudStW7f861elnG80l1vscdDaGiytrcdAS7UWexy0pdZi9tfWHq9Snr9ayi1mn2342M7hw4ene++9N+2///4pItK4cePy2xSz33ZE7hNPPJEqKirSIYcckv7yl7+k8ePHp6qqqlRRUZH/G6GYY6G1Wot5rnniiSdSZWVlyuVy6cc//nE666yzUnV1daqoqEiPP/540cdXW9Z1aW05xtqa297j4eGHH04RkYYOHVpwLvP444+X5TmmcV5HWe4Heiml9Nvf/jYNHDgwdenSJW2++eb5F4JS+mw4MHTo0ILtI2KZr4EDBxZs81//9V8p4rOLL7aWecEFF6QNN9wwdevWLfXs2TNtttlm6aKLLkqLFy/ObzN27Ni0zjrr5A/qXC6XBg0a1Gqt77//fqqtrW324t2jR49OK6+8cv736NGjR/rtb3/bYuYFF1yQNthgg1RbW5v69euXDjzwwPTmm2+WVOuLL76YvvKVr6Ta2trUs2fPtMcee6R//OMfBf/uwIEDm1z70047Lb/Ngw8+2OQ2DRetbur3+etf/5o222yz1KVLl7TmmmsucwA3lRcR6corr2xxnVrLPeyww/L73aqrrpq+8Y1vFPxxWEzmfvvtl/r165eqq6tT//7908iRIwuGzcVkNrjzzjvTRhttlGpqatJ66623zD5VbPZ9992XIiJNnz59mfuKyZw3b14aNWpUGjBgQOratWtaa6210tixYwuekIrJvfHGG9Naa62VunTpkurr69PRRx+d3n///RYzzz333Bb3xyVLlqTTTjst1dfXp5qamrT99tun5557riBj6NChBRddb20fv/LKK1s9TpbOHD16dOrTp0+zuQ2f093U14MPPthsbkqf/RHcXO5HH32U9tprr9S/f//UpUuX1K9fv7T77rsvc8Hc9q7B0pp6om9vrcXm7r333s1mfvjhh2nYsGFp1VVXTdXV1WnAgAHpe9/7XnrjjTfKUuvll1+e1llnndS1a9e06aabLjPwLGZdf/e736Xa2tqCfb/UWmfNmpUOOeSQ1L9//9S1a9e07rrrpl//+tf5ix1/nrktPV4ppXT++een1VdfPf94/fSnPy3oLU1ltuU5pJhe0Jbc9vaD5jKPPvrolFIquhe0VmtH5RbTY1pbg6a0pRe0Vmsx/aCttba3F7Rl32pvbnOZjf/ntZTa32PaUmsxvaAjctuyBsX0mMMOOyytuuqqKZfLpYjPLsL+X//1X/n7i+kvrWUWc66RUko77rhjqqqqShGfXQNl4403zp/7lnKu0RG5G264YaqsrEwRkXK5XOrVq1c677zzUkrF9Za21NqUtvSX1h6vYjIb/ofUhrWqrq5OQ4YMyddayjlMW9agvX2rLWvQ3szW1qBBe/tWa7UWe17U2roWk9vScZBScT0rpZT++Mc/pnXXXTdVV1en9dZbL91yyy1p6NChBQO9YnrX0tm1tbVpxIgRJWWOHj06DRgwIP83+7Bhw5YZPhbbE5urtZR+uPTabrjhhvlaS+ldLeU2pa1/fzW3BqX0mLbU2t5+0FKtDYr5O6m1WovtBy3VWsxxu99++xW8brjSSisV9IJij9dtt90232OqqqrSVlttVfD6VbHHVku5xR4Hra3B0tp6DLRUa7HHQVtqLWZ/be3xKnZ/bS232OeaU089NX9Nsy5duqQjjzyy4P5i99uOyN19991TdXV1/jl8gw02SFOmTMnfX+yx0FKtxT7X3HnnnWmNNdbIn8fU19fnr+lYyvNMa+u6tLYeY23Jbe/x0PA69qBBgwrOZRqU+hyzdF5HyKX0/z5nEQAAAAAAAMic5foaegAAAAAAAPBFZ6AHAAAAAAAAGWagBwAAAAAAABlmoAcAAAAAAAAZZqAHAAAAAAAAGWagBwAAAAAAABlmoAcAAAAAAAAZZqAHAAAAAAAAGWagBwAAQJOmTJkSlZWVscsuu3R2KQAAACu0XEopdXYRAAAAZM/3v//9WGmlleL3v/99vPjiizFgwIDOLgkAAGCF5B16AAAALOODDz6Im266KX70ox/FrrvuGldddVXB/XfccUcMHjw4amtrY8cdd4yrr746crlcvP/++/ltpkyZEttvv33U1tbGGmusEccdd1x88MEHn+8vAgAA8AVgoAcAAMAybrzxxlh33XVj3XXXjYMOOiiuvPLKaPiAl3/961+xzz77xJ577hnTpk2LI444IsaOHVvw888991wMHz48Ro4cGc8++2zceOON8cgjj8QxxxzTGb8OAADAcs1HbgIAALCM7bbbLvbdd98YNWpUfPrpp9GvX7+44YYbYqeddopTTz017r777njuuefy2//0pz+N008/Pd5777340pe+FN/97nejtrY2fve73+W3eeSRR2Lo0KHxwQcfRNeuXTvj1wIAAFgueYceAAAABaZPnx5PPPFE7L///hERUVVVFfvtt19cccUV+fu33HLLgp/56le/WvD9008/HVdddVWstNJK+a/hw4fHkiVLYsaMGZ/PLwIAAPAFUdXZBQAAAJAtl19+eXz66aex2mqr5W9LKUV1dXW89957kVKKXC5X8DNLf/jLkiVL4ogjjojjjjtumfwBAwZ0TOEAAABfUAZ6AAAA5H366adxzTXXxK9//esYNmxYwX177713XHfddbHeeuvFPffcU3DfU089VfD95ptvHi+88EKss846HV4zAADAF51r6AEAAJB3++23x3777Rdz5syJurq6gvvGjh0b99xzT9x6662x7rrrxvHHHx+HH354TJs2LU488cR488034/3334+6urp49tlnY+utt45DDz00fvCDH0T37t3jpZdeikmTJsVvfvObTvrtAAAAlk+uoQcAAEDe5ZdfHjvttNMyw7yIz96hN23atHjvvffi5ptvjltvvTU22WSTuPjii2Ps2LEREVFTUxMREZtssklMnjw5Xnnllfj6178em222WfzsZz+Lfv36fa6/DwAAwBeBd+gBAABQstNPPz0uueSSmDlzZmeXAgAA8IXjGnoAAAC020UXXRRbbrll9O7dOx599NE466yz4phjjunssgAAAL6QDPQAAABot1deeSV++ctfxn/+858YMGBAnHjiiTFmzJjOLgsAAOALyUduAgAAAAAAQIZVdHYBAAAAAAAAQPMM9AAAAAAAACDDDPQAAAAAAAAgwwz0AAAAAAAAIMMM9AAAAAAAACDDDPQAAAAAAAAgwwz0AAAAAAAAIMMM9AAAAAAAACDDDPQAAAAAAAAgw/4v8o5bvcVyFD8AAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 2200x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#checking the whitch age reported more fraud:\n",
    "data=df.groupby('Age').agg({'Survived':'count'}).reset_index()\n",
    "\n",
    "fig, (ax1) = plt.subplots(1,1,figsize=(22, 6))\n",
    "graph =sns.barplot(x='Age', y='Survived', data = data, ax=ax1)\n",
    "\n",
    "graph.set_xticklabels(graph.get_xticklabels(),\n",
    "                    rotation=0,\n",
    "                    horizontalalignment='right'\n",
    "                    );"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "85bf47b3",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:48.499280Z",
     "iopub.status.busy": "2023-08-08T14:50:48.498837Z",
     "iopub.status.idle": "2023-08-08T14:50:48.895615Z",
     "shell.execute_reply": "2023-08-08T14:50:48.893925Z"
    },
    "papermill": {
     "duration": 0.424683,
     "end_time": "2023-08-08T14:50:48.899867",
     "exception": false,
     "start_time": "2023-08-08T14:50:48.475184",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: ylabel='Survived'>"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "df.groupby(['Pclass']).count().plot(kind='pie', y='Survived', autopct='%1.0f%%')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "a210bd19",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:48.964526Z",
     "iopub.status.busy": "2023-08-08T14:50:48.963714Z",
     "iopub.status.idle": "2023-08-08T14:50:48.972000Z",
     "shell.execute_reply": "2023-08-08T14:50:48.971155Z"
    },
    "papermill": {
     "duration": 0.034083,
     "end_time": "2023-08-08T14:50:48.974052",
     "exception": false,
     "start_time": "2023-08-08T14:50:48.939969",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    266\n",
       "1    152\n",
       "Name: Survived, dtype: int64"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Survived'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "22267188",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:49.021410Z",
     "iopub.status.busy": "2023-08-08T14:50:49.020666Z",
     "iopub.status.idle": "2023-08-08T14:50:49.047886Z",
     "shell.execute_reply": "2023-08-08T14:50:49.046529Z"
    },
    "papermill": {
     "duration": 0.054504,
     "end_time": "2023-08-08T14:50:49.051249",
     "exception": false,
     "start_time": "2023-08-08T14:50:48.996745",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Importing the required libraries to perform pre-processing in train data and splitting into train-test split:\n",
    "from sklearn import preprocessing\n",
    "from sklearn.impute import SimpleImputer\n",
    "\n",
    "from sklearn.compose import ColumnTransformer\n",
    "\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "8d9665bb",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:49.098701Z",
     "iopub.status.busy": "2023-08-08T14:50:49.098242Z",
     "iopub.status.idle": "2023-08-08T14:50:49.114112Z",
     "shell.execute_reply": "2023-08-08T14:50:49.112495Z"
    },
    "papermill": {
     "duration": 0.042153,
     "end_time": "2023-08-08T14:50:49.116530",
     "exception": false,
     "start_time": "2023-08-08T14:50:49.074377",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 418 entries, 0 to 417\n",
      "Data columns (total 12 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  418 non-null    int64  \n",
      " 1   Survived     418 non-null    int64  \n",
      " 2   Pclass       418 non-null    int64  \n",
      " 3   Name         418 non-null    object \n",
      " 4   Sex          418 non-null    object \n",
      " 5   Age          418 non-null    float64\n",
      " 6   SibSp        418 non-null    int64  \n",
      " 7   Parch        418 non-null    int64  \n",
      " 8   Ticket       418 non-null    object \n",
      " 9   Fare         418 non-null    float64\n",
      " 10  Cabin        91 non-null     object \n",
      " 11  Embarked     418 non-null    object \n",
      "dtypes: float64(2), int64(5), object(5)\n",
      "memory usage: 39.3+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "33a1ec77",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:49.164350Z",
     "iopub.status.busy": "2023-08-08T14:50:49.163582Z",
     "iopub.status.idle": "2023-08-08T14:50:49.169583Z",
     "shell.execute_reply": "2023-08-08T14:50:49.168533Z"
    },
    "papermill": {
     "duration": 0.032635,
     "end_time": "2023-08-08T14:50:49.172079",
     "exception": false,
     "start_time": "2023-08-08T14:50:49.139444",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "df.drop([\"PassengerId\",\"Name\",\"Cabin\",\"Ticket\"], axis = 1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "00f4f22c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:49.219266Z",
     "iopub.status.busy": "2023-08-08T14:50:49.218797Z",
     "iopub.status.idle": "2023-08-08T14:50:49.234032Z",
     "shell.execute_reply": "2023-08-08T14:50:49.232876Z"
    },
    "papermill": {
     "duration": 0.041622,
     "end_time": "2023-08-08T14:50:49.236536",
     "exception": false,
     "start_time": "2023-08-08T14:50:49.194914",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>34.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.8292</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>47.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.0000</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>male</td>\n",
       "      <td>62.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>9.6875</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.6625</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Survived  Pclass     Sex   Age  SibSp  Parch    Fare Embarked\n",
       "0         0       3    male  34.5      0      0  7.8292        Q\n",
       "1         1       3  female  47.0      1      0  7.0000        S\n",
       "2         0       2    male  62.0      0      0  9.6875        Q\n",
       "3         0       3    male  27.0      0      0  8.6625        S"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head(4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "669f6e29",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:49.285367Z",
     "iopub.status.busy": "2023-08-08T14:50:49.284564Z",
     "iopub.status.idle": "2023-08-08T14:50:49.291051Z",
     "shell.execute_reply": "2023-08-08T14:50:49.290201Z"
    },
    "papermill": {
     "duration": 0.033334,
     "end_time": "2023-08-08T14:50:49.293479",
     "exception": false,
     "start_time": "2023-08-08T14:50:49.260145",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Dropping the target variable from train data and storing the rest of the attributes in \"X\" and target attribute in \"y\":\n",
    "X = df.drop([\"Survived\"], axis = 1)\n",
    "y = df[\"Survived\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "0ee32ff2",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:49.341739Z",
     "iopub.status.busy": "2023-08-08T14:50:49.340922Z",
     "iopub.status.idle": "2023-08-08T14:50:49.347346Z",
     "shell.execute_reply": "2023-08-08T14:50:49.346063Z"
    },
    "papermill": {
     "duration": 0.033128,
     "end_time": "2023-08-08T14:50:49.349697",
     "exception": false,
     "start_time": "2023-08-08T14:50:49.316569",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(418, 7) (418,)\n"
     ]
    }
   ],
   "source": [
    "# printing the shapes of X and y: \n",
    "print(X.shape, y.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "c6770aae",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:49.397624Z",
     "iopub.status.busy": "2023-08-08T14:50:49.397241Z",
     "iopub.status.idle": "2023-08-08T14:50:49.405545Z",
     "shell.execute_reply": "2023-08-08T14:50:49.404479Z"
    },
    "papermill": {
     "duration": 0.035448,
     "end_time": "2023-08-08T14:50:49.408086",
     "exception": false,
     "start_time": "2023-08-08T14:50:49.372638",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123,stratify=y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "3e711d82",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:49.458885Z",
     "iopub.status.busy": "2023-08-08T14:50:49.458477Z",
     "iopub.status.idle": "2023-08-08T14:50:49.463991Z",
     "shell.execute_reply": "2023-08-08T14:50:49.463083Z"
    },
    "papermill": {
     "duration": 0.032798,
     "end_time": "2023-08-08T14:50:49.466710",
     "exception": false,
     "start_time": "2023-08-08T14:50:49.433912",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(334, 7)\n",
      "(84, 7)\n",
      "(334,)\n",
      "(84,)\n"
     ]
    }
   ],
   "source": [
    "#checking shape of train and test data:\n",
    "print(X_train.shape)\n",
    "print(X_test.shape)\n",
    "print(y_train.shape)\n",
    "print(y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "f8dfc2a5",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:49.516735Z",
     "iopub.status.busy": "2023-08-08T14:50:49.515277Z",
     "iopub.status.idle": "2023-08-08T14:50:49.525603Z",
     "shell.execute_reply": "2023-08-08T14:50:49.524525Z"
    },
    "papermill": {
     "duration": 0.038237,
     "end_time": "2023-08-08T14:50:49.528189",
     "exception": false,
     "start_time": "2023-08-08T14:50:49.489952",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Survived      2\n",
       "Pclass        3\n",
       "Sex           2\n",
       "Age          79\n",
       "SibSp         7\n",
       "Parch         8\n",
       "Fare        169\n",
       "Embarked      3\n",
       "dtype: int64"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.nunique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "073b89b9",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:49.577490Z",
     "iopub.status.busy": "2023-08-08T14:50:49.576798Z",
     "iopub.status.idle": "2023-08-08T14:50:49.588806Z",
     "shell.execute_reply": "2023-08-08T14:50:49.587680Z"
    },
    "papermill": {
     "duration": 0.039609,
     "end_time": "2023-08-08T14:50:49.591422",
     "exception": false,
     "start_time": "2023-08-08T14:50:49.551813",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "df[\"Pclass\"] = df[\"Pclass\"].astype('category')\n",
    "df[\"Sex\"] = df[\"Sex\"].astype('category')\n",
    "df[\"Embarked\"] = df[\"Embarked\"].astype('category')\n",
    "df[\"Survived\"] = df[\"Survived\"].astype('category')\n",
    "X_train[\"Pclass\"] = X_train[\"Pclass\"].astype('category')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "ad642520",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:49.640218Z",
     "iopub.status.busy": "2023-08-08T14:50:49.639804Z",
     "iopub.status.idle": "2023-08-08T14:50:49.648352Z",
     "shell.execute_reply": "2023-08-08T14:50:49.647177Z"
    },
    "papermill": {
     "duration": 0.036162,
     "end_time": "2023-08-08T14:50:49.650810",
     "exception": false,
     "start_time": "2023-08-08T14:50:49.614648",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Pclass      category\n",
       "Sex           object\n",
       "Age          float64\n",
       "SibSp          int64\n",
       "Parch          int64\n",
       "Fare         float64\n",
       "Embarked      object\n",
       "dtype: object"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "89ae59dc",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:49.700353Z",
     "iopub.status.busy": "2023-08-08T14:50:49.699892Z",
     "iopub.status.idle": "2023-08-08T14:50:49.710246Z",
     "shell.execute_reply": "2023-08-08T14:50:49.709004Z"
    },
    "papermill": {
     "duration": 0.037892,
     "end_time": "2023-08-08T14:50:49.712673",
     "exception": false,
     "start_time": "2023-08-08T14:50:49.674781",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Pclass      0\n",
       "Sex         0\n",
       "Age         0\n",
       "SibSp       0\n",
       "Parch       0\n",
       "Fare        0\n",
       "Embarked    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_test.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a4d18f3f",
   "metadata": {
    "papermill": {
     "duration": 0.024292,
     "end_time": "2023-08-08T14:50:49.760591",
     "exception": false,
     "start_time": "2023-08-08T14:50:49.736299",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "a6e2f816",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:49.809902Z",
     "iopub.status.busy": "2023-08-08T14:50:49.809485Z",
     "iopub.status.idle": "2023-08-08T14:50:49.818438Z",
     "shell.execute_reply": "2023-08-08T14:50:49.817229Z"
    },
    "papermill": {
     "duration": 0.036174,
     "end_time": "2023-08-08T14:50:49.820560",
     "exception": false,
     "start_time": "2023-08-08T14:50:49.784386",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Age', 'SibSp', 'Parch', 'Fare'], dtype='object')"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#defining numerical columns in data:\n",
    "numeric_columns=df.select_dtypes(['float64','int64']).columns\n",
    "numeric_columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "f4f5456c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:49.869746Z",
     "iopub.status.busy": "2023-08-08T14:50:49.869326Z",
     "iopub.status.idle": "2023-08-08T14:50:49.877677Z",
     "shell.execute_reply": "2023-08-08T14:50:49.876612Z"
    },
    "papermill": {
     "duration": 0.035617,
     "end_time": "2023-08-08T14:50:49.879908",
     "exception": false,
     "start_time": "2023-08-08T14:50:49.844291",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Pclass', 'Sex', 'Embarked'], dtype='object')"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#defining categorical columns in data:\n",
    "cat_columns=X_train.select_dtypes(['object','category']).columns\n",
    "cat_columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "b7c27f38",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:49.929531Z",
     "iopub.status.busy": "2023-08-08T14:50:49.929127Z",
     "iopub.status.idle": "2023-08-08T14:50:49.933679Z",
     "shell.execute_reply": "2023-08-08T14:50:49.932521Z"
    },
    "papermill": {
     "duration": 0.032387,
     "end_time": "2023-08-08T14:50:49.936041",
     "exception": false,
     "start_time": "2023-08-08T14:50:49.903654",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Importing the required preprocessing libraries:\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "10df0db6",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:49.986687Z",
     "iopub.status.busy": "2023-08-08T14:50:49.986076Z",
     "iopub.status.idle": "2023-08-08T14:50:50.005191Z",
     "shell.execute_reply": "2023-08-08T14:50:50.004242Z"
    },
    "papermill": {
     "duration": 0.04773,
     "end_time": "2023-08-08T14:50:50.007809",
     "exception": false,
     "start_time": "2023-08-08T14:50:49.960079",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "#standardscaler for numerical columns\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train[numeric_columns])\n",
    "\n",
    "X_train_num = pd.DataFrame(scaler.transform(X_train[numeric_columns]), columns=numeric_columns)\n",
    "X_test_num = pd.DataFrame(scaler.transform(X_test[numeric_columns]), columns=numeric_columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "4911d4de",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:50.058186Z",
     "iopub.status.busy": "2023-08-08T14:50:50.057400Z",
     "iopub.status.idle": "2023-08-08T14:50:50.066047Z",
     "shell.execute_reply": "2023-08-08T14:50:50.065136Z"
    },
    "papermill": {
     "duration": 0.036493,
     "end_time": "2023-08-08T14:50:50.068481",
     "exception": false,
     "start_time": "2023-08-08T14:50:50.031988",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "#ohe = OneHotEncoder()\n",
    "ohe = OneHotEncoder(handle_unknown='ignore', drop = \"first\")\n",
    "ohe.fit(X_train[cat_columns])\n",
    "\n",
    "columns_ohe = list(ohe.get_feature_names_out(cat_columns))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "cb8b79f2",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:50.118303Z",
     "iopub.status.busy": "2023-08-08T14:50:50.117856Z",
     "iopub.status.idle": "2023-08-08T14:50:50.128636Z",
     "shell.execute_reply": "2023-08-08T14:50:50.127314Z"
    },
    "papermill": {
     "duration": 0.038855,
     "end_time": "2023-08-08T14:50:50.131407",
     "exception": false,
     "start_time": "2023-08-08T14:50:50.092552",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "X_train_cat = ohe.transform(X_train[cat_columns])\n",
    "X_test_cat  = ohe.transform(X_test[cat_columns])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "d606bed9",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:50.181286Z",
     "iopub.status.busy": "2023-08-08T14:50:50.180852Z",
     "iopub.status.idle": "2023-08-08T14:50:50.190516Z",
     "shell.execute_reply": "2023-08-08T14:50:50.189431Z"
    },
    "papermill": {
     "duration": 0.037693,
     "end_time": "2023-08-08T14:50:50.193031",
     "exception": false,
     "start_time": "2023-08-08T14:50:50.155338",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "X_train_cat = pd.DataFrame(X_train_cat.todense(), columns=columns_ohe)\n",
    "X_test_cat  = pd.DataFrame(X_test_cat.todense(), columns=columns_ohe)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "03022176",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:50.243251Z",
     "iopub.status.busy": "2023-08-08T14:50:50.242476Z",
     "iopub.status.idle": "2023-08-08T14:50:50.249675Z",
     "shell.execute_reply": "2023-08-08T14:50:50.248533Z"
    },
    "papermill": {
     "duration": 0.035112,
     "end_time": "2023-08-08T14:50:50.252129",
     "exception": false,
     "start_time": "2023-08-08T14:50:50.217017",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "X_train = pd.concat([X_train_num, X_train_cat], axis=1)\n",
    "X_test = pd.concat([X_test_num, X_test_cat], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "6cfe5998",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:50.301988Z",
     "iopub.status.busy": "2023-08-08T14:50:50.301605Z",
     "iopub.status.idle": "2023-08-08T14:50:50.306543Z",
     "shell.execute_reply": "2023-08-08T14:50:50.305244Z"
    },
    "papermill": {
     "duration": 0.032919,
     "end_time": "2023-08-08T14:50:50.308892",
     "exception": false,
     "start_time": "2023-08-08T14:50:50.275973",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "7000f45e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:50.360911Z",
     "iopub.status.busy": "2023-08-08T14:50:50.360524Z",
     "iopub.status.idle": "2023-08-08T14:50:50.365806Z",
     "shell.execute_reply": "2023-08-08T14:50:50.364595Z"
    },
    "papermill": {
     "duration": 0.034975,
     "end_time": "2023-08-08T14:50:50.368014",
     "exception": false,
     "start_time": "2023-08-08T14:50:50.333039",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "logistic_model = LogisticRegression(solver='liblinear',random_state=1230)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "731ddbd9",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:50.418352Z",
     "iopub.status.busy": "2023-08-08T14:50:50.417887Z",
     "iopub.status.idle": "2023-08-08T14:50:50.432031Z",
     "shell.execute_reply": "2023-08-08T14:50:50.430486Z"
    },
    "papermill": {
     "duration": 0.043399,
     "end_time": "2023-08-08T14:50:50.435389",
     "exception": false,
     "start_time": "2023-08-08T14:50:50.391990",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "Model=logistic_model.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "1a0633ae",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:50.487422Z",
     "iopub.status.busy": "2023-08-08T14:50:50.486958Z",
     "iopub.status.idle": "2023-08-08T14:50:50.501648Z",
     "shell.execute_reply": "2023-08-08T14:50:50.499939Z"
    },
    "papermill": {
     "duration": 0.043715,
     "end_time": "2023-08-08T14:50:50.504228",
     "exception": false,
     "start_time": "2023-08-08T14:50:50.460513",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "y_train_Pred = logistic_model.predict(X_train)\n",
    "y_test_Pred = logistic_model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "79590baa",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:50.565608Z",
     "iopub.status.busy": "2023-08-08T14:50:50.565212Z",
     "iopub.status.idle": "2023-08-08T14:50:50.572939Z",
     "shell.execute_reply": "2023-08-08T14:50:50.571880Z"
    },
    "papermill": {
     "duration": 0.044642,
     "end_time": "2023-08-08T14:50:50.576059",
     "exception": false,
     "start_time": "2023-08-08T14:50:50.531417",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "y_pred = logistic_model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "ef31d813",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:50.630541Z",
     "iopub.status.busy": "2023-08-08T14:50:50.630122Z",
     "iopub.status.idle": "2023-08-08T14:50:50.640402Z",
     "shell.execute_reply": "2023-08-08T14:50:50.639217Z"
    },
    "papermill": {
     "duration": 0.038343,
     "end_time": "2023-08-08T14:50:50.642701",
     "exception": false,
     "start_time": "2023-08-08T14:50:50.604358",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[53  0]\n",
      " [ 0 31]]\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix, accuracy_score\n",
    "cmr = confusion_matrix(y_test, y_pred)\n",
    "print(cmr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "b77af83e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:50.692807Z",
     "iopub.status.busy": "2023-08-08T14:50:50.692403Z",
     "iopub.status.idle": "2023-08-08T14:50:50.700949Z",
     "shell.execute_reply": "2023-08-08T14:50:50.699829Z"
    },
    "papermill": {
     "duration": 0.036204,
     "end_time": "2023-08-08T14:50:50.703048",
     "exception": false,
     "start_time": "2023-08-08T14:50:50.666844",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1.0"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "logistic_m=accuracy_score(y_test,y_pred)\n",
    "logistic_m"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "56e60219",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:50.753626Z",
     "iopub.status.busy": "2023-08-08T14:50:50.752934Z",
     "iopub.status.idle": "2023-08-08T14:50:50.765312Z",
     "shell.execute_reply": "2023-08-08T14:50:50.763930Z"
    },
    "papermill": {
     "duration": 0.0403,
     "end_time": "2023-08-08T14:50:50.767750",
     "exception": false,
     "start_time": "2023-08-08T14:50:50.727450",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training score:  1.0\n",
      "Testing score:  1.0\n"
     ]
    }
   ],
   "source": [
    "print('Training score: ', round(logistic_model.score(X_train, y_train),3))\n",
    "print('Testing score: ', round(logistic_model.score(X_test, y_test),3))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "e6520567",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:50.818339Z",
     "iopub.status.busy": "2023-08-08T14:50:50.817349Z",
     "iopub.status.idle": "2023-08-08T14:50:50.871727Z",
     "shell.execute_reply": "2023-08-08T14:50:50.870502Z"
    },
    "papermill": {
     "duration": 0.082756,
     "end_time": "2023-08-08T14:50:50.874581",
     "exception": false,
     "start_time": "2023-08-08T14:50:50.791825",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier\n",
    "\n",
    "dt = DecisionTreeClassifier(criterion= 'entropy', max_depth=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "32a10d75",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:50.927262Z",
     "iopub.status.busy": "2023-08-08T14:50:50.926820Z",
     "iopub.status.idle": "2023-08-08T14:50:50.944914Z",
     "shell.execute_reply": "2023-08-08T14:50:50.944021Z"
    },
    "papermill": {
     "duration": 0.047245,
     "end_time": "2023-08-08T14:50:50.947177",
     "exception": false,
     "start_time": "2023-08-08T14:50:50.899932",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>DecisionTreeClassifier(criterion=&#x27;entropy&#x27;, max_depth=10)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">DecisionTreeClassifier</label><div class=\"sk-toggleable__content\"><pre>DecisionTreeClassifier(criterion=&#x27;entropy&#x27;, max_depth=10)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "DecisionTreeClassifier(criterion='entropy', max_depth=10)"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dt.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "88b69506",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:50.999163Z",
     "iopub.status.busy": "2023-08-08T14:50:50.998377Z",
     "iopub.status.idle": "2023-08-08T14:50:51.007673Z",
     "shell.execute_reply": "2023-08-08T14:50:51.006441Z"
    },
    "papermill": {
     "duration": 0.038072,
     "end_time": "2023-08-08T14:50:51.010237",
     "exception": false,
     "start_time": "2023-08-08T14:50:50.972165",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "y_train_predt = dt.predict(X_train)\n",
    "y_test_predt = dt.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "195de8d0",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:51.062064Z",
     "iopub.status.busy": "2023-08-08T14:50:51.060955Z",
     "iopub.status.idle": "2023-08-08T14:50:51.072854Z",
     "shell.execute_reply": "2023-08-08T14:50:51.071590Z"
    },
    "papermill": {
     "duration": 0.040593,
     "end_time": "2023-08-08T14:50:51.075549",
     "exception": false,
     "start_time": "2023-08-08T14:50:51.034956",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training score:  1.0\n",
      "Testing score:  1.0\n"
     ]
    }
   ],
   "source": [
    "print('Training score: ', round(dt.score(X_train, y_train),3))\n",
    "print('Testing score: ', round(dt.score(X_test, y_test),3))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "403f2de6",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:51.127520Z",
     "iopub.status.busy": "2023-08-08T14:50:51.126629Z",
     "iopub.status.idle": "2023-08-08T14:50:51.134606Z",
     "shell.execute_reply": "2023-08-08T14:50:51.133603Z"
    },
    "papermill": {
     "duration": 0.036288,
     "end_time": "2023-08-08T14:50:51.136768",
     "exception": false,
     "start_time": "2023-08-08T14:50:51.100480",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1.0"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy_score(y_test, y_test_predt)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "82909c24",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:51.189160Z",
     "iopub.status.busy": "2023-08-08T14:50:51.188400Z",
     "iopub.status.idle": "2023-08-08T14:50:51.195080Z",
     "shell.execute_reply": "2023-08-08T14:50:51.193949Z"
    },
    "papermill": {
     "duration": 0.035576,
     "end_time": "2023-08-08T14:50:51.197652",
     "exception": false,
     "start_time": "2023-08-08T14:50:51.162076",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "final_pred = dt.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "a4ac56f3",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:51.249857Z",
     "iopub.status.busy": "2023-08-08T14:50:51.249427Z",
     "iopub.status.idle": "2023-08-08T14:50:51.255651Z",
     "shell.execute_reply": "2023-08-08T14:50:51.254798Z"
    },
    "papermill": {
     "duration": 0.034969,
     "end_time": "2023-08-08T14:50:51.257746",
     "exception": false,
     "start_time": "2023-08-08T14:50:51.222777",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,\n",
       "       0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0,\n",
       "       0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1,\n",
       "       1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1])"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "final_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "11099a29",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:51.309968Z",
     "iopub.status.busy": "2023-08-08T14:50:51.309530Z",
     "iopub.status.idle": "2023-08-08T14:50:51.314763Z",
     "shell.execute_reply": "2023-08-08T14:50:51.313483Z"
    },
    "papermill": {
     "duration": 0.033922,
     "end_time": "2023-08-08T14:50:51.316912",
     "exception": false,
     "start_time": "2023-08-08T14:50:51.282990",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "final_pred=pd.DataFrame(final_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "fb102194",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-08-08T14:50:51.371189Z",
     "iopub.status.busy": "2023-08-08T14:50:51.370379Z",
     "iopub.status.idle": "2023-08-08T14:50:51.380363Z",
     "shell.execute_reply": "2023-08-08T14:50:51.379397Z"
    },
    "papermill": {
     "duration": 0.039728,
     "end_time": "2023-08-08T14:50:51.382519",
     "exception": false,
     "start_time": "2023-08-08T14:50:51.342791",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   0\n",
       "0  0\n",
       "1  0\n",
       "2  1"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "final_pred.head(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cfc6ad09",
   "metadata": {
    "papermill": {
     "duration": 0.027235,
     "end_time": "2023-08-08T14:50:51.437665",
     "exception": false,
     "start_time": "2023-08-08T14:50:51.410430",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 20.746576,
   "end_time": "2023-08-08T14:50:52.486232",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2023-08-08T14:50:31.739656",
   "version": "2.4.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
