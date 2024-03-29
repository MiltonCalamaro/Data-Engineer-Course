import logging
# logging.basicConfig (level = logging.INFO)
import subprocess
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s, %(levelname)s %(name)s, %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("scrapper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("__name__")
news_sites_uids = ['eluniversal','elpais']

def main():
	_extract()
	_transform()
	_load()
	
def _extract():
	logger.info('Starting extract process')
	for news_site_uid in news_sites_uids:
		subprocess.run(['python','main.py', news_site_uid], cwd = './extract')
		subprocess.run(['cp', '{}_articles.csv'.format(news_site_uid), '../transform/{}_.csv'.format(news_site_uid)], cwd = './extract')
		
def _transform():
	logger.info('Starting transform process')
	for news_site_uid in news_sites_uids: 
		dirty_data_filename = '{}_.csv'.format(news_site_uid)
		clean_data_filename = 'clean_{}'.format(dirty_data_filename)
		subprocess.run(['python','main.py',dirty_data_filename], cwd = './transform')
		subprocess.run(['rm',dirty_data_filename], cwd = './transform')
		subprocess.run(['cp',clean_data_filename, '../load/{}.csv'.format(news_site_uid)], cwd = './transform')
		
def _load():
	logger.info('Starting load process')
	for news_site_uid in news_sites_uids:
		clean_data_filename = '{}.csv'.format(news_site_uid)
		subprocess.run(['python','main.py',clean_data_filename], cwd = './load')
		subprocess.run(['rm',clean_data_filename],cwd = './load')


if __name__ == '__main__':
	main()
