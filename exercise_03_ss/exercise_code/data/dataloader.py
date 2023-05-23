"""Definition of Dataloader"""

import numpy as np


class DataLoader:
    """
    Dataloader Class
    Defines an iterable batch-sampler over a given dataset
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        """
        :param dataset: dataset from which to load the data
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        ########################################################################
        # TODO:                                                                #
        # Define an iterable function that samples batches from the dataset.   #
        # Each batch should be a dict containing numpy arrays of length        #
        # batch_size (except for the last batch if drop_last=True)             #
        # Hints:                                                               #
        #   - np.random.permutation(n) can be used to get a list of all        #
        #     numbers from 0 to n-1 in a random order                          #
        #   - To load data efficiently, you should try to load only those      #
        #     samples from the dataset that are needed for the current batch.  #
        #     An easy way to do this is to build a generator with the yield    #
        #     keyword, see https://wiki.python.org/moin/Generators             #
        #   - Have a look at the "DataLoader" notebook first. This function is #
        #     supposed to combine the functions:                               #
        #       - combine_batch_dicts                                          #
        #       - batch_to_numpy                                               #
        #       - build_batch_iterator                                         #
        #     in section 1 of the notebook.                                    #
        ########################################################################

            

        # build datasets
        def build_loder():
            batches = build_batches(
                batch_size=self.batch_size
            )
            combined_batches = [combine_batch_dicts(batch) for batch in batches]
            numpy_batches = [batch_to_numpy(batch) for batch in combined_batches]
            return numpy_batches

        # build batches
        def build_batches(batch_size):
            batches = []  # list of all mini-batches
            batch = []  # current mini-batch
            if self.shuffle:
                for i in np.random.permutation(len(self.dataset)):
                    batch.append(self.dataset[i])
                    if len(batch) == batch_size:  # if the current mini-batch is full,
                        batches.append(batch)  # add it to the list of mini-batches,
                        batch = []
            else:
                for i in range(len(self.dataset)):
                    batch.append(self.dataset[i])
                    if len(batch) == batch_size:  # if the current mini-batch is full,
                        batches.append(batch)  # add it to the list of mini-batches,
                        batch = []  # and start a new mini-batch
                # and start a new mini-batch
            if self.drop_last:
                return batches
            elif batch:
                batches.append(batch)
                return batches
        

        # combine batch dicts
        def combine_batch_dicts(batch):
            batch_dict = {}
            for data_dict in batch:
                for key, value in data_dict.items():
                    if key not in batch_dict:
                        batch_dict[key] = []
                    batch_dict[key].append(value)
            return batch_dict

        # batch to numpy
        def batch_to_numpy(batch):
            numpy_batch = {}
            for key, value in batch.items():
                numpy_batch[key] = np.array(value)
            return numpy_batch

        ###  -----

        batch = []
        batches = build_loder()

        for batch in batches:
            yield batch
        
        if self.shuffle:
            pass

        if self.drop_last:
            pass

        '''
        def combine_batch_dicts(batch):
            batch_dict = {}
            for data_dict in batch:
                for key, value in data_dict.items():
                    if key not in batch_dict:
                        batch_dict[key] = []
                    batch_dict[key].append(value)
            return batch_dict

        # batch to numpy
        def batch_to_numpy(batch):
            numpy_batch = {}
            for key, value in batch.items():
                numpy_batch[key] = np.array(value)
            return numpy_batch


        batches = []  # list of all mini-batches
        batch = []  # current mini-batch
        order = range(len(self.dataset)) 
        if self.shuffle:
            order = np.random.permutation(len(self.dataset))
        for i in order:
            batch.append(self.dataset[i])
            if len(batch) == self.batch_size:  # if the current mini-batch is full,
                batches.append(batch)  # add it to the list of mini-batches,
                batch = []  # and start a new mini-batch
        if not self.drop_last:
            if batch:
                batches.append(batch)
        elif batch:
            batches.append(batch)

        combined_batches = [combine_batch_dicts(batch) for batch in batches]
        numpy_batches = [batch_to_numpy(batch) for batch in combined_batches]

        for batch in batches:
            yield numpy_batches     
        '''

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def __len__(self):
        length = None
        ########################################################################
        # TODO:                                                                #
        # Return the length of the dataloader                                  #
        # Hint: this is the number of batches you can sample from the dataset. #
        # Don't forget to check for drop last!                                 #
        ########################################################################

        length = int(len(self.dataset) / self.batch_size)

        if not self.drop_last:
            if len(self.dataset) % self.batch_size:
                length += 1

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return length
