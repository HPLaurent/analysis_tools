import numpy as np
import numpy.lib.recfunctions
import os
import time
import glob

from utils_const import *



class fp_data:
    """
    Class for fire patches manipulation 
    Perform simple reduction operations on the data when required (remove missing data, select land cover, select patches following mask ID and field conversion  ...).

    attributes: fp_data, _survey, _cut_off, _type_fp, _flag_fp

    """

    def __init__(self, survey='MODIS_6', cut_off=5,type_fp='',flag_fp='',temp_rep='/home/orchidee01/plaurent/products/speedrun/final/temp_database/'):
        """
        Database constructor : type_fp represent the level of information of the data (land cover, tree cover, FRP). Flag allow to load directly a pre-selected version of the db (faster).

        """

        self._temp_rep = temp_rep

        if (flag_fp != ''):
            root_dir_fp = self._temp_rep
        else:
            root_dir_fp = '/home/orchidee01/plaurent/products/speedrun/final/'

        if (type_fp == 'FRP'):
            root_dir_fp += 'FRP/'
            self._temp_rep += 'FRP/'

        name_file_db = 'fire_patches_'+survey+'_final_co_'+str(cut_off) + flag_fp+ '.npy'

        if (os.path.exists(root_dir_fp + name_file_db)==False):
            raise IOError(root_dir_fp + name_file_db + 'does not exist')

        self.data = np.load(root_dir_fp + name_file_db)
        self._type_fp = type_fp
        self._survey = survey
        self._cut_off = cut_off
        self._flag_fp = flag_fp
        self._name_file_db = name_file_db
        self.size = self.data.size


        if (flag_fp==''):
            year_min = np.where(self.data['MAX_BD'] < self.data['MIN_BD'], self.data['YEAR']-1,self.data['YEAR'])
            year_mean = np.where(self.data['MAX_BD'] < self.data['MEAN_BD'], self.data['YEAR']-1,self.data['YEAR'])
            self.data = np.lib.recfunctions.rec_append_fields(self.data,['YEAR_MIN','YEAR_MAX','YEAR_MEAN'],[year_min,self.data['YEAR'],year_mean])


    def __str__(self):
        return 'List of ' + str(self.data.size)+ ' fire patches for ' + self._survey + ' (cut off = ' + str(self._cut_off) + ')'

        
    def __add__(self,other_fp):
        return np.concatenate((self.data,other_fp))


    def day_frp2julian(self,save=False):
        """
        Convert list_days from MCD14ML date format to Julian day (1-366)

        """

        flag_op = '_bd_converted'

        for field_bd in ['MIN_BD_FRP','MAX_BD_FRP']:
            print self.data.size
            bd_frp = [time.strptime(str(int(self.data[field_bd][day])), "%Y%m%d") for day in np.arange(self.data.size)]
            bd_frp_day = np.asarray([bd_frp[day].tm_yday for day in np.arange(self.data.size)])
            self.data[field_bd][:] = bd_frp_day[:]

        self._flag_fp += flag_op

        if save:
            self.save_to_temp()


    def day_frp2month(self,save=False):
        """
        Convert list_days from MCD14ML date format to Month (1-12)

        """

        flag_op = '_bd_converted'

        for field_bd in ['MIN_BD_FRP']:
            print self.data.size
            bd_frp = [time.strptime(str(int(self.data[field_bd][day])), "%Y%m%d") for day in np.arange(self.data.size)]
            bd_frp_day = np.asarray([bd_frp[day].tm_mon for day in np.arange(self.data.size)])
            self.data[field_bd][:] = bd_frp_day[:]

        self._flag_fp += flag_op

        if save:
            self.save_to_temp()


    def select_patch_with_frp(self,save=False):
        """
        Remove patches without FRP information

        """

        flag_op = '_frp_only'

        if (self._type_fp == 'FRP'):
            self.data = self.data[(self.data['MEAN_FRP'] != 0.) & (self.data['MIN_BD_FRP'] > 0)  & (self.data['MAX_BD_FRP'] > 0)]
        else:
            raise TypeError('Fire patch database does not contain FRP information, cannot perform operation ...')

        self._flag_fp += flag_op
            
        if save:
            self.save_to_temp()


    def select_year(self,year,save=False):
        """
        Select patches that have burned during this year.

        """

        flag_op = '_year_' + str(year)

        self.data = self.data[(self.data['YEAR_MIN'] == year) & (self.data['YEAR_MAX'] == year)]

        self._flag_fp += flag_op
            
        if save:
            self.save_to_temp()


    def select_random(self,ratio=0.01,save=False):
        """
        Get all fire patches belonging to GFED_id areas. GFED_id can either be a integer or a string (see get_GFED_labels())

        """
        print int(1./ratio)
        self.data = self.data[np.random.randint(int(1./ratio), size=self.data.size) == 1]

        flag_op = '_ratio_' + str(ratio)
        self._flag_fp +=  flag_op

        if save:
            print self._flag_fp
            self.save_to_temp()


    def remove_cropfires(self,save=False):
        """
        Remove all patches overlapping crops

        """

        flag_op = '_no_crop_fires'

        if (self._type_fp in ['FRP','land_cover']):
            self.data = self.data[(self.data['LAND_COV_TYPE_1'] > 35)]
        else:
            raise TypeError('Fire patch database does not contain land cover information, cannot perform operation ...')

        self._flag_fp +=  flag_op

        if save:
            print self._flag_fp
            self.save_to_temp()



    def get_fp(self):
        """
        Get all fire patches (encapsulation)

        """
        return np.copy(self.data)


    def get_fp_GFED(self,GFED_id):
        """
        Get all fire patches belonging to GFED_id areas. GFED_id can either be a integer or a string (see get_GFED_labels())

        """

        label_reduced,label_full = get_GFED_labels()
        if (type(GFED_id) is int):
            return np.copy(self.data[(self.data['GFED_MASK'] == GFED_id+1)])
        elif (type(GFED_id) is str):
            ID = np.arange(0,len(label_reduced))[np.asarray(label_reduced) == GFED_id][0]+1
            return np.copy(self.data[(self.data['GFED_MASK'] == ID)])


    def get_fp_window(self,min_lon,min_lat,max_lon=None,max_lat=None,width=None,height=None):        
        """
        Get all fire patches belonging to the window min_lon,max_lon,min_lat,max_lat.

        """

        if (width is not None):
            max_lon = min_lon+width
        if (height is not None):
            max_lat = min_lat+height
        print min_lon,max_lon,min_lat,max_lat
        return np.copy(self.data[(self.data['CENTER_Y'] <= max_lat) & (self.data['CENTER_Y'] > min_lat) & (self.data['CENTER_X'] <= max_lon) & (self.data['CENTER_X'] > min_lon)])


    def print_available_temp_db(self):
        """
        Get all available temporary fire patch databases.

        """

        all_name_db =  glob.glob('/home/orchidee01/plaurent/products/speedrun/final/temp_database/*')
        for name in all_name_db:
            print name


    def save_to_temp(self):
        """
        Save temporary fire patch database.

        """

        np.save(self._temp_rep+'fire_patches_'+self._survey+'_final_co_'+str(self._cut_off) + self._flag_fp+ '.npy',self.data)
