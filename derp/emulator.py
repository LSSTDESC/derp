"""
The ``derp`` :py:class:`Emulator` is the main class in the ``derp`` package.
"""
import healpy
import numpy as np
import pandas as pd
import GCRCatalogs
import FoFCatalogMatching

class Emulator(object):
    """
    Worker object, that can construct a training set from an input truth
    catalog matched to an output DRP Object catalog, use it to train a 
    generative model, and then use that trained model to predict a new, 
    "emulated" DRP catalog given a further input truth catalog.
    """
    def __init__(self):
        # Initialize attributes:
        self.X = None
        self.y = None
        self.Nts = None
        return
    
    def __repr__(self):
        """Returns representation of the emulator object"""
        s = "derp.Emulator"
        if self.Nts is not None:
            s += ", containing {}-object training set".format(self.Nts)
            s += " to support prediction of {} DRP object attributes".format(self.X.shape[1])
            s += " given {} input (true) object attributes.".format(self.y.shape[1])
        return s
    
    def make_training_set(self, truth=None, 
                          observed=None, 
                          region=None, 
                          true_quantities=None, 
                          observed_quantities=None):
        """
        Construct a design matrix ``X`` and array of response variables
        ``y`` from a given truth catalog and DRP Object table.
        
        Parameters
        ----------
        truth: str
            Name of the truth catalog (for GCR)
        observed: str
            Name of the observed DRP object catalog (for GCR)
        region: tuple, float
            RA, DEC, radius of sky region to use objects from
        true_quantities: list, str
            List of quantity names to use from the truth table
        observed_quantities=None
            List of quantity names to use from the observed object table
            
        Returns
        -------
        X: :py:obj:`pandas.Dataframe`
            Design matrix, true properties of one-to-one matched objects
        y: :py:obj:`pandas.Dataframe`
            Response variables, observed properties of one-to-one matches
        """
        # Set up filters:
        center_ra, center_dec, radius = region
        ra_min, ra_max = center_ra - radius, center_ra + radius
        dec_min, dec_max = center_dec - radius, center_dec + radius
        coord_filters = ['ra >= {}'.format(ra_min),
                         'ra < {}'.format(ra_max),
                         'dec >= {}'.format(dec_min),
                         'dec < {}'.format(dec_max)]
        list_of_healpix = setup_filter_on_healpix(region)
        mag_filters = [(np.isfinite, 'mag_r_cModel'),
                       ('mag_r_cModel < 27.0')]


        # Load data:
        truth_catalog = GCRCatalogs.load_catalog(truth, {'md5': None})
        true_objects = truth_catalog.get_quantities(true_quantities,
                           native_filters=['r<27.0',
                                           'healpix_2048<=%d' % list_of_healpix.max(),
                                           'healpix_2048>=%d' % list_of_healpix.min()],
                           filters=[(lambda hp: filter_on_healpix(hp, list_of_healpix), 'healpix_2048')]+coord_filters)      
        observed_catalog = GCRCatalogs.load_catalog(observed)
        observed_objects = observed_catalog.get_quantities(observed_quantities, 
                               filters=(mag_filters + coord_filters))

        # FoF matching:
        results = FoFCatalogMatching.match(catalog_dict={'truth': true_objects,
                                                         'observed': observed_objects},
                                           linking_lengths=1.0,
                                           catalog_len_getter=lambda x: len(x['ra']))
        
        # Get one-to-one matches:
        truth_mask = results['catalog_key'] == 'truth'
        observed_mask = ~truth_mask

        n_groups = results['group_id'].max() + 1
        n_true = np.bincount(results['group_id'][truth_mask], minlength=n_groups)
        n_observed = np.bincount(results['group_id'][observed_mask], minlength=n_groups)

        one_to_one_group_mask = np.in1d(results['group_id'], 
                                        np.flatnonzero((n_true == 1) & 
                                                       (n_observed == 1)))
        truth_idx = results['row_index'][one_to_one_group_mask & truth_mask]
        observed_idx = results['row_index'][one_to_one_group_mask & observed_mask]
        self.X = pd.DataFrame(true_objects).iloc[truth_idx].reset_index(drop=True)
        self.y = pd.DataFrame(observed_objects).iloc[observed_idx].reset_index(drop=True)
        
        self.Nts = self.X.shape[0]
        
        return self.X, self.y

    
def setup_filter_on_healpix(region):
    """
    Return a list of healpix ids overlapping the given sky region
    
    Parameters
    ----------
    region: tuple, float
        RA, DEC, radius of sky region to use objects from
        
    Returns
    -------
    list_of_healpix: list, int
    """
    center_ra, center_dec, radius = region
    center_ra_rad = np.radians(center_ra)
    center_dec_rad = np.radians(center_dec)
    center_vec = np.array([np.cos(center_dec_rad)*np.cos(center_ra_rad),
                           np.cos(center_dec_rad)*np.sin(center_ra_rad),
                           np.sin(center_dec_rad)])
    list_of_healpix = healpy.query_disc(2048, center_vec, np.radians(radius), 
                                        nest=True, inclusive=True)
    return list_of_healpix


def filter_on_healpix(hp, list_of_healpix):
    return np.array([hh in list_of_healpix for hh in hp])
