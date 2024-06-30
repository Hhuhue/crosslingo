from invoke import Collection

import data_processing


#data_processing.create_word_index.pre(data_processing.extract_word_frequencies)
ns = Collection('tasks')

ns.add_task(data_processing.extract_word_frequencies, name='frequency')
ns.add_task(data_processing.create_word_index, name='index')
ns.add_task(data_processing.train_word2vec, name='train')
ns.add_task(data_processing.test_word2vec, name='test')

if __name__ == "__main__":
    import invoke
    invoke.run()