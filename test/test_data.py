


def test_column_names(data):
    """
    Checks for column names and also the order
    :param data:
    :return: null
    """
    expected_columns = ['age',
                        'workclass',
                        'fnlgt',
                        'education',
                        'education-num',
                        'marital-status',
                        'occupation',
                        'relationship',
                        'race',
                        'sex',
                        'capital-gain',
                        'capital-loss',
                        'hours-per-week',
                        'native-country',
                        'salary']

    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_columns) == list(these_columns)


def test_row_count(data):
    """
    Asserts the rows of the dataset is between 15000 and 1000000
    """
    assert 15000 < data.shape[0] < 1000000


def test_duplicates(data):
    """
   Check if there are duplicates
   :param data:
   :return:
   """

    assert data.shape == data.dropna().shape



