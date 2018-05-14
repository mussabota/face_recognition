from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base


engine = create_engine('sqlite:///users_data.sqlite')


db_session = scoped_session(sessionmaker(bind=engine))


Base = declarative_base()
Base.query = db_session.query_property()


class Users(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    first_name = Column(String(50))
    last_name = Column(String(50))
    user_id = Column(String(10))
    last_entered = Column(String(100))

    def __init__(self, first_name=None, last_name=None, user_id=None, last_entered=None):
        self.first_name = first_name
        self.last_name = last_name
        self.user_id = user_id
        self.last_entered = last_entered

    def __repr__(self):
        return '<User {} {} {}>'.format(self.first_name, self.last_name, self.user_id)


if __name__ == '__main__':
    Base.metadata.create_all(bind=engine)
