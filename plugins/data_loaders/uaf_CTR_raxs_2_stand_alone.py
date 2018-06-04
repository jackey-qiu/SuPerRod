''' <h1>gsecars ctr data loader </h1>
Loads the data from whitespace seperated column formatted ascii data files.
It is intended for surface x-ray diffraction data where the data sets consists
of rod scans along the l-direction (perpendicular to the surface). The plugin
sorts each rod with equal h and k values into one data sets. The l-direction
is also sorted. <p>
The default columns are the following:<br>
1st column X values (l value for CTR and E for RAXS); 2nd column h values; 3rd values k values;
4th column Y (l value for RAXS but a column wont be used for CTR); 5th column intensity; and the 6th column The standard deviation of the intensities;
7th column L of first Bragg peak, 8th column L spacing of Bragg peaks
 The other settings are just as in the default data loader.<p>

The h,k values is stored as extra data in data.extra_data dictonary as
h and k.
'''

import numpy as np
import data

class Plugin():
    def __init__(self):
        self.X_col=0
        self.h_col = 1
        self.k_col = 2
        self.Y_col = 3
        self.I_col = 4
        self.eI_col =5
        self.LB_col = 6
        self.dL_col = 7
        self.comment = '#'
        self.skip_rows = 0
        self.delimiter = None
        self.data=data.DataList()

    def LoadData(self, filename):
        '''LoadData(self, data_item_number, filename) --> none

        Loads the data from filename into the data_item_number.
        '''
        try:
            load_array = np.loadtxt(filename, delimiter = self.delimiter,
                comments = self.comment, skiprows = self.skip_rows)
        except Exception, e:
            ShowWarningDialog(self.parent, 'Could not load the file: ' +\
                    filename + ' \nPlease check the format.\n\n numpy.loadtxt'\
                    + ' gave the following error:\n'  +  str(e))
        else:
            # For the freak case of only one data point
            if len(load_array.shape) < 2:
                load_array = np.array([load_array])
            # Check so we have enough columns
            if load_array.shape[1]-1 < max(self.X_col,self.h_col, self.k_col,\
                     self.Y_col, self.I_col, self.eI_col, self.LB_col, self.dL_col):
                ShowWarningDialog(self.parent, 'The data file does not contain'\
                        + 'enough number of columns. It has ' + str(load_array.shape[1])\
                        + ' columns. Rember that the column index start at zero!')
                # Okay now we have showed a dialog lets bail out ...
                return
            # The data is set by the default Template.__init__ function, neat hu
            # Note that the loaded data goes into *_raw so that they are not
            # changed by the transforms

            # Create an record array so we can sort the data properly
            data = np.rec.fromarrays([\
                     load_array[:,self.X_col],\
                     load_array[:,self.h_col].round().astype(type(1)),\
                     load_array[:,self.k_col].round().astype(type(1)),\
                     load_array[:,self.Y_col], load_array[:,self.I_col],\
                     load_array[:,self.eI_col],\
                     load_array[:,self.LB_col], load_array[:,self.dL_col]\
                    ],\
                     names = 'X, h, k, Y, I, eI, LB, dL')
            # Sort the data
            data.sort(order = ('h','k','Y','X'))
            i = 0
            while i < len(data):
                # Find all the data for each rod
                tmp = data.compress(np.bitwise_and(np.bitwise_and(data['h'] == data[i]['h'],\
                                    data['k'] == data[i]['k']),data['Y'] == data[i]['Y']))
                self.data.add_new('(%i, %i, %3.2f)'%(tmp['h'][0], tmp['k'][0], tmp['Y'][0]))
                self.data[-1].x_raw = tmp['X']
                self.data[-1].y_raw =tmp['I']
                self.data[-1].error_raw = tmp['eI']
                # Run the commands on the data - this also sets the x,y, error memebers
                # of that data item.
                self.data[-1].run_command()
                self.data[-1].set_extra_data('h', tmp['h'], 'h')
                self.data[-1].set_extra_data('k', tmp['k'], 'k')
                self.data[-1].set_extra_data('Y', tmp['Y'], 'Y')
                self.data[-1].set_extra_data('LB', tmp['LB'], 'LB')
                self.data[-1].set_extra_data('dL', tmp['dL'], 'dL')
                # Increase the index
                i += len(tmp)
