from sqlalchemy import Column, String, Integer

from base import Base

class Article(Base):
	__tablename__ = 'articles'
	id = Column(String,primary_key = True)
	body = Column(String)
	host = Column(String)
	newspaper_uid = Column(String)
	n_tokens_body = Column(Integer)
	n_tokens_title = Column(Integer)
	title = Column(String)
	url =Column(String, unique = True)
	def __init__(self,
				id,
				body,
				host,
				newspaper_uid,
				n_tokens_body,
				n_tokens_title,
				title,
				url):
		self.id = id
		self.body = body
		self.host = host
		self.newspaper_uid = newspaper_uid
		self.n_tokens_body = n_tokens_body
		self.n_tokens_title = n_tokens_title
		self.title = title
		self.url = url