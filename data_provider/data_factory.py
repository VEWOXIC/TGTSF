from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_TGTSF, Dataset_TGTSF_elec, Dataset_TGTSF_weather
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader

def custom_collate_fn(batch): # pad all the news to the same length on batch dimension
    batch_x = [torch.tensor(item[0]) for item in batch]
    batch_y = [torch.tensor(item[1]) for item in batch]
    batch_news = [item[2] for item in batch]
    batch_des = [item[3] for item in batch]

    max_len = max([tensor.shape[1] for tensor in batch_news])
    # print('padding to', max_len)
    masks = []
    
    for i in range(len(batch_news)):
        mask = torch.cat((torch.zeros((batch_news[i].shape[0], batch_news[i].shape[1])), torch.ones((batch_news[i].shape[0], max_len - batch_news[i].shape[1]))), dim=1)
        batch_news[i] = torch.cat((batch_news[i], torch.zeros((batch_news[i].shape[0], max_len - batch_news[i].shape[1], batch_news[i].shape[2]))), dim=1)
        #  create a mask matrix for the news. set it as 1 where all the elements on dimension 3 of the batch_news are all 0
        # mask = (batch_news[i].sum(dim=2) == 0)
        masks.append(mask)
        
    
        
    # x,y 不需要pad?
    padded_x = pad_sequence(batch_x, batch_first=True, padding_value=0)
    padded_y = pad_sequence(batch_y, batch_first=True, padding_value=0)
    padded_news = pad_sequence(batch_news, batch_first=True, padding_value=0)
    padded_des = pad_sequence(batch_des, batch_first=True, padding_value=0)
    mask = pad_sequence(masks, batch_first=True, padding_value=0)
    return padded_x, padded_y, padded_news, padded_des, mask


def TGTSF_data_provider(args, flag, text_encoder, data_path=None, news_path=None, des_path=None, global_norm=False):
    if data_path is None:
        data_path = args.data_path       
    if news_path is None:
        news_path = args.news_path
    if des_path is None:
        des_path = args.des_path

    # check if any of the path is a list
    try:
        assert isinstance(data_path, str)
        assert isinstance(news_path, str)
        assert isinstance(des_path, str)
    except AssertionError:
        print('data_path should be a string, to load multiple datasets, use TGTSF_pretrain_data_provider')
        exit()

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size//4

    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size

    print('des_pre'+str(args.des_pre_embed))
    print('news_pre'+str(args.news_pre_embed))

    data_set = Dataset_TGTSF(
        root_path=args.root_path,
        data_path=data_path,
        news_path=news_path,
        des_path=des_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        info_overhead=args.info_overhead,
        news_pre_embed=args.news_pre_embed,
        des_pre_embed=args.des_pre_embed,
        text_encoder=text_encoder,
        add_date=args.add_date,
        text_dim=args.text_dim,
        scale=(not global_norm), # disable the individual norm
        
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=custom_collate_fn)
    return data_set, data_loader

def TGTSF_elec_data_provider(args, flag, text_encoder, data_path=None, news_path=None, des_path=None, global_norm=False):
    if data_path is None:
        data_path = args.data_path       
    if news_path is None:
        news_path = args.news_path
    if des_path is None:
        des_path = args.des_path

    # check if any of the path is a list
    try:
        assert isinstance(data_path, str)
        assert isinstance(news_path, str)
        assert isinstance(des_path, str)
    except AssertionError:
        print('data_path should be a string, to load multiple datasets, use TGTSF_pretrain_data_provider')
        exit()

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size//4

    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size

    print('des_pre'+str(args.des_pre_embed))
    print('news_pre'+str(args.news_pre_embed))

    data_set = Dataset_TGTSF_elec(
        root_path=args.root_path,
        data_path=data_path,
        news_path=news_path,
        des_path=des_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        info_overhead=args.info_overhead,
        news_pre_embed=args.news_pre_embed,
        des_pre_embed=args.des_pre_embed,
        text_encoder=text_encoder,
        add_date=args.add_date,
        text_dim=args.text_dim,
        scale=(not global_norm), # disable the individual norm
        
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=custom_collate_fn)
    return data_set, data_loader

def TGTSF_weather_data_provider(args, flag, text_encoder=None, data_path=None, news_path=None, des_path=None, global_norm=False):
    if data_path is None:
        data_path = args.data_path       
    if news_path is None:
        news_path = args.news_path
    if des_path is None:
        des_path = args.des_path

    # check if any of the path is a list
    try:
        assert isinstance(data_path, str)
        assert isinstance(news_path, str)
        assert isinstance(des_path, str)
    except AssertionError:
        print('data_path should be a string, to load multiple datasets, use TGTSF_pretrain_data_provider')
        exit()

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size

    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size

    print('des_pre'+str(args.des_pre_embed))
    print('news_pre'+str(args.news_pre_embed))

    data_set = Dataset_TGTSF_weather(
        root_path=args.root_path,
        data_path=data_path,
        news_path=news_path,
        des_path=des_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        info_overhead=args.info_overhead,
        news_pre_embed=args.news_pre_embed,
        des_pre_embed=args.des_pre_embed,
        text_encoder=text_encoder,
        add_date=args.add_date,
        text_dim=args.text_dim,
        scale=(not global_norm),
        stride = args.stride # disable the individual norm
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=custom_collate_fn)
    return data_set, data_loader


def TGTSF_pretrain_data_provider(args, flag, text_encoder, global_norm=False):

    assert len(args.data_path) == len(args.news_path) == len(args.des_path)

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size

    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    print('des_pre'+str(args.des_pre_embed))
    print('news_pre'+str(args.news_pre_embed))

    datasets = []


    for data_path, news_path, des_path in zip(args.data_path, args.news_path, args.des_path):
        data_set = Dataset_TGTSF(
            root_path=args.root_path,
            data_path=data_path,
            news_path=news_path,
            des_path=des_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            info_overhead=args.info_overhead,
            news_pre_embed=args.news_pre_embed,
            des_pre_embed=args.des_pre_embed,
            text_encoder=text_encoder,
            add_date=args.add_date,
            text_dim=args.text_dim,
            scale=(not global_norm), # disable the individual norm
        )
        datasets.append(data_set)
        print(flag, len(data_set))

    data_set = torch.utils.data.ConcatDataset(datasets)
    print(f'total data {len(data_set)}')
    # calculate the mean and std of the dataset

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=custom_collate_fn)
    return data_set, data_loader
