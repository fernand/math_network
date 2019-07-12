import dataset; ds = dataset.get_datasets('/home/fernand/math/data', 'test'); batch=[ds[0], ds[1]]
print(dataset.collate_data(batch))
