"""
数据集下载和准备脚本
用于下载公开数据集，并转换为项目所需的 CSV 格式
"""

import os
import pandas as pd
from datasets import load_dataset

# 数据目录
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def download_imdb_dataset():
    """
    下载 IMDB 电影评论数据集（情感分类）
    - 标签：0=负面, 1=正面
    """
    print("正在下载 IMDB 数据集...")
    
    try:
        dataset = load_dataset("imdb")
        
        # 转换为 DataFrame
        train_df = pd.DataFrame({
            "text": dataset["train"]["text"],
            "target": dataset["train"]["label"]
        })
        
        test_df = pd.DataFrame({
            "text": dataset["test"]["text"],
            "target": dataset["test"]["label"]
        })
        
        # 保存为 CSV
        train_path = os.path.join(DATA_DIR, "imdb_train.csv")
        test_path = os.path.join(DATA_DIR, "imdb_test.csv")
        
        train_df.to_csv(train_path, index=False, encoding="utf-8")
        test_df.to_csv(test_path, index=False, encoding="utf-8")
        
        print(f"✅ IMDB 训练集已保存: {train_path} ({len(train_df)} 条)")
        print(f"✅ IMDB 测试集已保存: {test_path} ({len(test_df)} 条)")
        
        return train_path, test_path
        
    except Exception as e:
        print(f"❌ 下载 IMDB 数据集失败: {e}")
        return None, None


def download_sst2_dataset():
    """
    下载 SST-2 情感分析数据集
    - 标签：0=负面, 1=正面
    """
    print("正在下载 SST-2 数据集...")
    
    try:
        dataset = load_dataset("glue", "sst2")
        
        train_df = pd.DataFrame({
            "text": dataset["train"]["sentence"],
            "target": dataset["train"]["label"]
        })
        
        val_df = pd.DataFrame({
            "text": dataset["validation"]["sentence"],
            "target": dataset["validation"]["label"]
        })
        
        train_path = os.path.join(DATA_DIR, "sst2_train.csv")
        val_path = os.path.join(DATA_DIR, "sst2_val.csv")
        
        train_df.to_csv(train_path, index=False, encoding="utf-8")
        val_df.to_csv(val_path, index=False, encoding="utf-8")
        
        print(f"✅ SST-2 训练集已保存: {train_path} ({len(train_df)} 条)")
        print(f"✅ SST-2 验证集已保存: {val_path} ({len(val_df)} 条)")
        
        return train_path, val_path
        
    except Exception as e:
        print(f"❌ 下载 SST-2 数据集失败: {e}")
        return None, None


def download_ag_news_dataset():
    """
    下载 AG News 新闻分类数据集
    - 标签：0=World, 1=Sports, 2=Business, 3=Sci/Tech
    """
    print("正在下载 AG News 数据集...")
    
    try:
        dataset = load_dataset("ag_news")
        
        train_df = pd.DataFrame({
            "text": dataset["train"]["text"],
            "target": dataset["train"]["label"]
        })
        
        test_df = pd.DataFrame({
            "text": dataset["test"]["text"],
            "target": dataset["test"]["label"]
        })
        
        train_path = os.path.join(DATA_DIR, "ag_news_train.csv")
        test_path = os.path.join(DATA_DIR, "ag_news_test.csv")
        
        train_df.to_csv(train_path, index=False, encoding="utf-8")
        test_df.to_csv(test_path, index=False, encoding="utf-8")
        
        print(f"✅ AG News 训练集已保存: {train_path} ({len(train_df)} 条)")
        print(f"✅ AG News 测试集已保存: {test_path} ({len(test_df)} 条)")
        
        return train_path, test_path
        
    except Exception as e:
        print(f"❌ 下载 AG News 数据集失败: {e}")
        return None, None


def download_chnsenticorp_dataset():
    """
    下载中文情感分析数据集 ChnSentiCorp
    - 标签：0=负面, 1=正面
    """
    print("正在下载 ChnSentiCorp 中文数据集...")
    
    try:
        # 尝试多种数据源
        dataset = None
        dataset_sources = [
            ("lansinuote/ChnSentiCorp", {}),  # 新版数据源
            ("c-s-ale/ChnSentiCorp", {}),     # 备选数据源
            ("seamew/ChnSentiCorp", {"trust_remote_code": True}),  # 旧版数据源需要信任代码
        ]
        
        for source, kwargs in dataset_sources:
            try:
                print(f"  尝试数据源: {source}")
                dataset = load_dataset(source, **kwargs)
                print(f"  ✓ 成功从 {source} 加载")
                break
            except Exception as e:
                print(f"  ✗ {source} 失败: {str(e)[:50]}...")
                continue
        
        if dataset is None:
            raise Exception("所有数据源都无法访问，尝试创建本地中文数据集")
        
        train_df = pd.DataFrame({
            "text": dataset["train"]["text"],
            "target": dataset["train"]["label"]
        })
        
        val_df = pd.DataFrame({
            "text": dataset["validation"]["text"],
            "target": dataset["validation"]["label"]
        })
        
        test_df = pd.DataFrame({
            "text": dataset["test"]["text"],
            "target": dataset["test"]["label"]
        })
        
        train_path = os.path.join(DATA_DIR, "chnsenticorp_train.csv")
        val_path = os.path.join(DATA_DIR, "chnsenticorp_val.csv")
        test_path = os.path.join(DATA_DIR, "chnsenticorp_test.csv")
        
        train_df.to_csv(train_path, index=False, encoding="utf-8")
        val_df.to_csv(val_path, index=False, encoding="utf-8")
        test_df.to_csv(test_path, index=False, encoding="utf-8")
        
        print(f"✅ ChnSentiCorp 训练集已保存: {train_path} ({len(train_df)} 条)")
        print(f"✅ ChnSentiCorp 验证集已保存: {val_path} ({len(val_df)} 条)")
        print(f"✅ ChnSentiCorp 测试集已保存: {test_path} ({len(test_df)} 条)")
        
        return train_path, val_path, test_path
        
    except Exception as e:
        print(f"❌ 下载 ChnSentiCorp 数据集失败: {e}")
        print("📝 正在创建本地中文情感数据集...")
        return create_chinese_sentiment_dataset()


def create_chinese_sentiment_dataset():
    """
    创建本地中文情感分析数据集（当在线数据集下载失败时使用）
    包含约200条中文评论数据
    """
    print("正在创建本地中文情感数据集...")
    
    # 正面评论 (target=1)
    positive_samples = [
        "这家酒店的服务真的太棒了，前台态度超好，房间干净整洁，下次还会再来！",
        "买了这款手机，性能超出预期，拍照效果一流，电池续航也很给力。",
        "这部电影剧情感人，演员演技在线，特效也很震撼，强烈推荐！",
        "餐厅的菜品味道鲜美，环境优雅，服务周到，是约会的好去处。",
        "这本书写得太好了，情节引人入胜，文笔优美，值得一读再读。",
        "客服态度非常好，问题解决得很及时，物流也很快，好评！",
        "这款护肤品用了一周，皮肤明显变好了，性价比很高。",
        "酒店位置绝佳，出行方便，早餐种类丰富，住得很舒心。",
        "产品质量很好，包装精美，送人很有面子，会回购的。",
        "这家店的服务态度一级棒，价格实惠，东西也很正宗。",
        "手机运行流畅，系统很稳定，外观设计也很漂亮。",
        "电影院环境很好，音效震撼，座椅舒适，观影体验极佳。",
        "这次旅行体验非常棒，导游专业负责，行程安排合理。",
        "商品和描述完全一致，发货速度快，包装也很用心。",
        "课程内容实用，老师讲解清晰，学到了很多知识。",
        "这款耳机音质太赞了，降噪效果一流，戴着很舒服。",
        "餐厅上菜速度快，分量足，口味正宗，性价比超高。",
        "酒店设施齐全，房间宽敞明亮，景观也很美。",
        "这个APP界面简洁，功能强大，用起来非常顺手。",
        "商家发货超快，东西质量也很好，非常满意这次购物。",
        "服务人员很专业，解答问题耐心细致，体验很好。",
        "产品做工精细，用料扎实，绝对是物超所值。",
        "这家餐厅的招牌菜太好吃了，下次还要带朋友来。",
        "快递小哥态度很好，送货上门很及时，好评！",
        "这款游戏画面精美，玩法有趣，让人爱不释手。",
        "酒店的早餐很丰盛，中西式都有，味道也不错。",
        "商品收到了，和卖家描述的一样，非常满意！",
        "这次售后服务体验很好，问题很快就解决了。",
        "产品设计很人性化，使用方便，真的是好物推荐。",
        "店家服务热情，商品质量好，价格也公道。",
        "这本书内容丰富，观点独到，读完收获很大。",
        "手机拍照效果惊艳，夜景模式特别出色。",
        "餐厅环境优美，菜品精致，是请客的好选择。",
        "产品包装严实，没有任何损坏，物流速度也快。",
        "这款洗面奶用着很温和，洗完脸不紧绑，好用！",
        "酒店服务周到，设施完善，住得非常舒适。",
        "商品质量超出预期，这个价格买到真的赚了。",
        "客服回复很及时，态度也很好，购物体验愉快。",
        "这部剧剧情紧凑，演员演技精湛，追剧停不下来。",
        "产品功能齐全，操作简单，老人家也能轻松使用。",
        "这家店的东西真的很实惠，品质也有保障。",
        "快递包装很仔细，商品完好无损，好评！",
        "餐厅的甜点超级好吃，环境也很适合拍照。",
        "这款产品真的太实用了，解决了我的大问题。",
        "服务态度五星好评，问题处理得又快又好。",
        "商品收到很惊喜，比图片还好看，推荐购买！",
        "酒店地理位置优越，交通便利，出行很方便。",
        "这个品牌的产品一直都很好用，会继续支持。",
        "课程设置合理，内容充实，学习效果明显。",
        "产品性能稳定，售后服务也很到位，很满意。",
    ]
    
    # 负面评论 (target=0)
    negative_samples = [
        "酒店房间太小了，隔音效果也差，晚上根本睡不好。",
        "这款手机用了一周就开始卡顿，电池也不耐用，太失望了。",
        "电影剧情拖沓，特效也很假，完全是浪费时间和钱。",
        "餐厅上菜超慢，等了一个多小时，菜还是凉的，差评！",
        "这本书内容空洞，毫无新意，完全是浪费钱。",
        "客服态度恶劣，问题一直没解决，再也不会来了。",
        "护肤品用了过敏，质量堪忧，不敢再用了。",
        "酒店卫生条件太差，床单都有污渍，太恶心了。",
        "产品和图片完全不符，质量也很差，被骗了。",
        "这家店服务态度极差，东西还贵，再也不来了。",
        "手机经常死机重启，系统bug太多，后悔买了。",
        "电影院座位太挤，空调也不好，体验很差。",
        "旅行团安排太紧凑，购物点倒是去了一大堆。",
        "商品有明显瑕疵，客服还不给退换，太气人了。",
        "课程内容水分大，老师照本宣科，浪费钱。",
        "耳机音质一般，降噪效果很差，不值这个价。",
        "餐厅分量太少，价格还贵，性价比极低。",
        "酒店设施老旧，空调声音很大，睡眠质量差。",
        "这个APP广告太多，还经常闪退，卸载了。",
        "商家发货慢，东西还破损了，售后态度差。",
        "服务人员态度冷淡，问什么都爱答不理的。",
        "产品做工粗糙，用料廉价，一点都不值。",
        "餐厅菜品味道一般，环境也很吵闹，不推荐。",
        "快递暴力运输，包裹都压变形了，太不负责了。",
        "游戏bug太多，还疯狂充值引导，差评！",
        "酒店早餐品种少，味道也不怎么样，失望。",
        "商品质量和描述不符，申请退款还很麻烦。",
        "售后服务太差了，打了好几次电话都没人处理。",
        "产品设计不合理，使用起来很不方便。",
        "店家态度傲慢，商品有问题还不承认。",
        "书的印刷质量很差，字迹模糊，不像正版。",
        "手机发热严重，玩一会儿就烫手，设计有问题。",
        "餐厅服务员态度差，上错菜了还不道歉。",
        "快递包装太简陋，商品都磕碰坏了。",
        "洗面奶用着刺激皮肤，不适合敏感肌。",
        "酒店隔音太差了，走廊说话都听得一清二楚。",
        "产品价格虚高，质量却很一般，不值得买。",
        "客服回复慢，还都是机器人回复，解决不了问题。",
        "这部剧剧情老套，演技尴尬，弃剧了。",
        "产品功能鸡肋，宣传的功能根本不实用。",
        "这家店东西贵不说，质量还没保障。",
        "快递延误严重，催了好几次都没用。",
        "餐厅甜点太甜腻了，吃不了几口就腻了。",
        "产品用了几天就坏了，质量也太差了吧。",
        "服务态度敷衍，问题拖了好久都不解决。",
        "商品图片和实物差距太大，严重货不对板。",
        "酒店位置偏僻，打车都不方便，选址太差。",
        "这个牌子以前还行，现在质量越来越差了。",
        "课程进度太快，根本跟不上，不适合零基础。",
        "产品售后形同虚设，坏了都没人管。",
    ]
    
    # 组合数据
    all_data = []
    for text in positive_samples:
        all_data.append({"text": text, "target": 1})
    for text in negative_samples:
        all_data.append({"text": text, "target": 0})
    
    df = pd.DataFrame(all_data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱顺序
    
    # 划分数据集 (70% 训练, 15% 验证, 15% 测试)
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size+val_size]
    test_df = df[train_size+val_size:]
    
    # 保存文件
    train_path = os.path.join(DATA_DIR, "chinese_sentiment_train.csv")
    val_path = os.path.join(DATA_DIR, "chinese_sentiment_val.csv")
    test_path = os.path.join(DATA_DIR, "chinese_sentiment_test.csv")
    
    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")
    
    print(f"✅ 中文情感训练集已保存: {train_path} ({len(train_df)} 条)")
    print(f"✅ 中文情感验证集已保存: {val_path} ({len(val_df)} 条)")
    print(f"✅ 中文情感测试集已保存: {test_path} ({len(test_df)} 条)")
    
    return train_path, val_path, test_path


def create_sample_dataset():
    """
    创建示例数据集（用于快速测试）
    """
    print("正在创建示例数据集...")
    
    sample_data = [
        # 正面评论
        {"text": "这部电影太棒了，剧情精彩，演员演技在线！", "target": 1},
        {"text": "非常好的产品，物超所值，强烈推荐！", "target": 1},
        {"text": "服务态度很好，下次还会再来！", "target": 1},
        {"text": "The movie was absolutely fantastic, great acting!", "target": 1},
        {"text": "Excellent product, exactly what I needed.", "target": 1},
        {"text": "Amazing experience, highly recommended!", "target": 1},
        {"text": "质量很好，发货速度快，满意！", "target": 1},
        {"text": "Great service and fast delivery!", "target": 1},
        {"text": "这家餐厅的菜品味道一绝，环境也很棒！", "target": 1},
        {"text": "The best purchase I've made this year!", "target": 1},
        
        # 负面评论
        {"text": "太失望了，完全不值这个价格。", "target": 0},
        {"text": "质量太差，用了一天就坏了。", "target": 0},
        {"text": "服务态度恶劣，再也不会来了。", "target": 0},
        {"text": "Terrible movie, waste of time and money.", "target": 0},
        {"text": "Poor quality, broke after one use.", "target": 0},
        {"text": "Awful experience, never coming back.", "target": 0},
        {"text": "包装破损，商品有瑕疵，很不满意。", "target": 0},
        {"text": "Shipping was slow and item was damaged.", "target": 0},
        {"text": "这餐厅又贵又难吃，环境也脏。", "target": 0},
        {"text": "Completely disappointed with this purchase.", "target": 0},
    ]
    
    df = pd.DataFrame(sample_data)
    
    # 划分训练集和测试集
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    
    train_path = os.path.join(DATA_DIR, "sample_train.csv")
    test_path = os.path.join(DATA_DIR, "sample_test.csv")
    
    train_df.to_csv(train_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")
    
    print(f"✅ 示例训练集已保存: {train_path} ({len(train_df)} 条)")
    print(f"✅ 示例测试集已保存: {test_path} ({len(test_df)} 条)")
    
    return train_path, test_path


def list_available_datasets():
    """列出可下载的数据集"""
    print("\n" + "="*50)
    print("可用的数据集下载选项：")
    print("="*50)
    print("1. IMDB     - 英文电影评论情感分类 (50,000条)")
    print("2. SST-2    - 英文情感分析 (67,000条)")
    print("3. AG News  - 英文新闻分类 (120,000条)")
    print("4. ChnSentiCorp - 中文情感分类 (在线下载)")
    print("5. 中文情感   - 中文情感分类 (本地生成, 100条)")
    print("6. Sample   - 示例数据集 (20条，用于快速测试)")
    print("7. All      - 下载所有数据集")
    print("="*50 + "\n")


if __name__ == "__main__":
    import sys
    
    list_available_datasets()
    
    if len(sys.argv) > 1:
        choice = sys.argv[1].lower()
    else:
        choice = input("请输入选项 (1-7): ").strip()
    
    if choice in ["1", "imdb"]:
        download_imdb_dataset()
    elif choice in ["2", "sst2"]:
        download_sst2_dataset()
    elif choice in ["3", "ag_news", "agnews"]:
        download_ag_news_dataset()
    elif choice in ["4", "chn", "chnsenticorp"]:
        download_chnsenticorp_dataset()
    elif choice in ["5", "chinese", "cn"]:
        create_chinese_sentiment_dataset()
    elif choice in ["6", "sample"]:
        create_sample_dataset()
    elif choice in ["7", "all"]:
        create_sample_dataset()
        create_chinese_sentiment_dataset()
        download_sst2_dataset()
        download_ag_news_dataset()
        download_chnsenticorp_dataset()
        download_imdb_dataset()
    else:
        print("无效选项，创建示例数据集...")
        create_sample_dataset()
    
    print("\n✅ 数据集准备完成！")
    print(f"📁 数据保存目录: {DATA_DIR}")
