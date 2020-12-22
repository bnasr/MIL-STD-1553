# Anomaly Detection for Cyber-Security Attacks

This repository prestents a `Python` module that has been designed to identify potential cyber attacks based on data from **MIL-STD-1553**.
MIL-STD-1553 is a military standard published by the United States Department of Defense that defines the mechanical, electrical, and functional characteristics of a serial data bus.  More information about the MIL-STD-1553 bus can be found from [this Wikipedia page](https://en.wikipedia.org/wiki/MIL-STD-1553).

![Sample MIL-STD-1553B Multiplex Data Bus Architecture](https://upload.wikimedia.org/wikipedia/commons/b/bf/MS1553B-Large-v2.png)




## Overview
At a very high level, the MIL-STD-1553 bus is a data transmission protocol that consists mainly of two types of computers:

- Bus Controllers (BC) 
- Remote Terminals (RT)

BCs control every action on the bus by dictating exactly when and how RTs communicate. 

Each 1553 message is composed of three parts:
- The "command word" - which is the BC message telling which RT to send/receive data, 
- The "status word" - the response from the RT acknowledging the BC command word, and
- The 0-32 "data" words - the useful information being passed across the bus. 

All three of these "words" are 16 bits in length:

* The 16-bit _**command**_ word include the addr, rxtx, subaddr, gap, and count fields:

- *addr* is the RT that the Bus Controller is addressing, 
- *rxtx* is a boolean that indicates whether the RT being addressed is supposed to be sending or transmitting the data field. If true, the RT is sending the data, and if false, the BC is sending the data, 
- *subaddr* is something akin to port numbers in IP communications (each RT could have several services running on it), 
- *count* is the number of 16-bit words sent in data, 
- *gap* (or inter message gap) is the time between messages in microseconds. 

* The _**data**_ word is the actual useful information being transmitted but in raw hex.
* The _**status**_ word is ignored and excluded from the dataset.


