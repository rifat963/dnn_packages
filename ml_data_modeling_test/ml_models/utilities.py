import pandas as pd


class utilities(object):

    def __init__(self):
        pass

    def resample_time_step(self, df,freq='H', column_name):
        """
        Resample dataset based on datetime values

        Args:
            1) dataset --> original dataset
            2) freq --> sampling freq:
                        non - No sampling
                        D - Daily
                        H - Hourly
                        T - Period: T - 1min, 2T- 2min, etc.
            3) column_name --> name the column with datetime values.

         Out:
            1) sample_df -> resampled dataset
        """
        if column_name:
            sample_df = df.set_index(pd.to_datetime(df[column_name]))
            sample_df = sample_df.drop([column_name], axis=1)
        else
            print('Select a column name with datetime for indexing')

        if freq == 'non':
            pass
        else:
            sample_df = sample_df.resample(freq).mean()
            # alternative way to sample the dataset using grouby
            # sample_df = sample_df.groupby(sample_df.index.to_period(freq)).mean()

        sample_df.dropna(inplace=True)

        return sample_df



