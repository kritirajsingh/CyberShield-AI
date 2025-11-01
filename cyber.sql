CREATE DATABASE cyber;
USE cyber;
alter user 'root'@'localhost' identified with 'mysql_native_password'by 'Dhanush@silver';
flush privileges;
CREATE TABLE users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(50) NOT NULL UNIQUE,
  email VARCHAR(100) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  create_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

SELECT id, username, email, password_hash FROM users;

DROP TABLE IF EXISTS crypto_wallets;

CREATE TABLE crypto_wallets(
wallet_id INT auto_increment PRIMARY KEY,
user_id INT NOT NULL,
wallet_address VARCHAR(255) unique NOT NULL,
balance DECIMAL(18,8) DEFAULT 0.0,
created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
FOREIGN KEY (user_id) references users(id) ON DELETE CASCADE
);

CREATE TABLE wallet_checks(
id INT auto_increment primary KEY,
user_id int not null,
wallet_id varchar(255) not null,
chain varchar(50) not null,
result_text text not null,
risk_level varchar(50) not null,
flagged tinyint(1) default 0,
notes text,
created_at timestamp default current_timestamp,
foreign key(user_id) references users(id) on delete cascade
);

show databases;