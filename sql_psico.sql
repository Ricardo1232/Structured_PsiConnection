use psiconnection;
select * from admin;
select * from paciente;
select * from practicante;
select * from encuesta;
select * from supervisor;
alter table paciente add column veriSurvey smallint;


UPDATE paciente SET sint_pri = NULL;
alter table paciente 
modify nombrePaci varchar(120),
modify apellidoPPaci varchar(120),
modify apellidoMPaci varchar(120);
delete from paciente where idPaci = 4;

alter table encuesta add column idSup int(11);
alter table paciente modify column sint_pri JSON;
CREATE TABLE horario (
    idHor INT(11) UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    fecha DATE NOT NULL,
    hora VARCHAR(5) NOT NULL,
    permitido BOOLEAN NOT NULL,
    practicante_id INT(11) UNSIGNED NOT NULL);
   
   
drop table horario;
 select * from horario;
 delete from horario;
 select * from citas;
 delete from citas where idCita >1897;
  delete from horario where idHor >1897;

    
select * from token;


delete from admin where idAd = 12;
delete from encuesta;
insert into horario(fecha,hora,permitido, practicante_id) values("2024-07-21","08:00",1,4);
create database Laravel;
use Laravel;
select * from candidatos;