drop table evenement;

create table if not exists evenement(
id INT NOT NULL PRIMARY KEY auto_increment,
type_evenement varchar(40),
nom varchar(40),
lieux varchar(40),
date_debut date,
duree int

);

insert into evenement values(1,'CES', 'salon', 'Las Vegas', '01/08/2018', 5);
insert into evenement values(2,'salon 1','salon',  'Paris', '02/12/2018', 2);
insert into evenement values(3,'salon 2', 'salon', 'Lyon', '03/24/2018', 3);
insert into evenement values(4,'salon 3', 'salon', 'Marseille', '04/15/2018', 2);
insert into evenement values(5,'salon 4', 'salon', 'Bordeaux', '05/02/2018', 5);
insert into evenement values(6,'dernier salon', 'salon', 'Lille', '06/21/2018', 2);


select * from evenemet;