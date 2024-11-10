---
layout: post
title: "Simple SMTP & POP3 servers"
date: 2022-11-10
summary: "A custom SMTP and POP3 server in Python, enabling email sending and retrieval through a simulated client-server environment"
keywords:
  ["TCP/IP", "SMTP", "POP3", "Sockets", "Relaying", "Protocols", "Python"]
categories: projects
---

**Authors:** Ibrahim El Kaddouri and Simon Desimpelaere

<br>

## Introduction

This article presents the detailed implementation of a POP3
(Post Office Protocol Version 3) and SMTP (Simple Mail Transfer Protocol)
server in Python for email management. The server enables users to retrieve
and manage their email messages. We discuss the motivation behind the project,
in-depth implementation details and handling of POP3 commands.

Email communication is an integral part of modern life and protocols like
POP3 play a vital role in email retrieval. The motivation behind this project
was to create a POP3/SMTP server from scratch to gain a deeper understanding of
email protocols and server development.

The primary objectives of this project include implementing a fully functional
POP3/SMTP server in Python, handling key POP3 commands, managing email messages,
ensuring robust error handling and providing documentation for reference.
We will learn how to use SMTP and POP3 to send and receive emails from a remote host.
SMTP is used to send mails from a mail client to the mail server,
while POP3 is used to access mail from a mail server via an email client.

## Protocols

A TCP connection is established between two SMTP servers to transmit
and receive emails. The sender establishes the connection and then delivers
a series of text commands to the recipient. There are additional SMTP details
in RFC 821. We will implement a subset of the SMTP command/replies in this project.

A POP3 server serves as the mail retrieval component in the email
communication process. It allows email clients to connect, authenticate
and retrieve messages from the user's mailbox.

As with SMTP, POP3 uses text commands over a TCP connection.
A sequence diagram of the protocol with more details is available in RFC 1939.

## Server Architecture

The mail server will host both the SMTP and POP3 servers. You can access
your mailbox using the POP3 client on your home client computer
and you can use the SMTP server to send emails using the SMTP client.

In the rest of this assignment, we will refer to the mail server machine as
MailServer and the remote machine from which you will access mail as the HomeClient.

All servers should support concurrent client connections. Consider scenarios in
which many users connect to the same mail server to access their mailboxes,
or numerous mail servers delivering mail to the same mail server simultaneously.

To visualise the different components, the following figure will help.
The sender and receiver are _both_ the HomeClient.
The Mail Server on the figure is the MailServer and hosts both the
POP3 and SMTP Servers.

<figure id="fig:diagram">
<img src="/assets/images/2022-11-10/SMTP_POP_graph.png" 
        style="width:110.0%" />
<figcaption> 
        <a href="https://www.suprsend.com/post/imap-vs-pop3-vs-smtp-choosing-the-right-email-protocol-ultimate-guide"> 
            source
        </a>
</figcaption>
</figure> <br>

## MailServer

The MailServer will do two things:

1. Run an SMTP mail server to accept and store emails from
   other mail clients or servers using SMTP.
2. Run a POP3 server to provide mailbox management and access from the HomeClient.

But first, we need to set up a few things. We will create a file called `userinfo.txt`
in the directory where the software will be run. On each line of this file,
there are one or more spaces between a user's login name and password.
Then, for each user in the same directory, there will be a subfolder
with the same name as the user. Each user's mailboxes will be kept in
the subdirectories that go with them.

```
.
├── userinfo.txt
├── USERS
│   ├── IbrahimEK
│   │   └── my_mailbox.txt
│   └── SimonDS
│       └── my_mailbox.txt
```

### SMTP mailserver

The program `mailserver_smtp.py` completes the first task above.
The program will accept the integer argument `my_port` on the command line
to provide the port. The received message is then added to a file called
`my_mailbox` in the user's subfolder when it has been received (remember
that the subdirectories are created manually). The mail address will contain the
username. The mail is attached in the manner described below:

```txt

From: <username>@<domain_name>
To: <username>@<domain_name>
Subject: <subject string, max 150 characters>
Received: <time at which received, in date : hour : minute>
<Message body – one or more lines>.

```

As a result, at any given time,
the `my_mailbox` file includes zero or more of these messages,
each of which is separated by a full stop (only the full
stop character in one line).

### POP3 mailserver

To do the second task above, a program called `popserver.py` will be the POP3
server. The program will accept the integer `POP3_port` as a command-line input
to specify the port on which the POP3 server will be running.

## HomeClient

On the HomeClient machine, sending and getting emails will be done by a program
called `mail_client.py`. The program will accept the `server_IP` address
command line option, which tells it where to connect to the SMTP and POP3
servers (i.e., the IP address of the MailServer machine). Before saving
the login and password locally, the program will first ask for them.
After that, it will ask the user to choose from the following three options:

-- **Mail Sending:** enables sending emails by the user  
-- **Mail Management:** includes retrieving/deleting mail, etc.  
-- **Exit:** the application

## Mail Sending

The user should type the message precisely as it is shown below:

```txt
From: <username>@<domain name>
To: <username>@<domain name>
Subject: <subject string, max 150 characters>
<Message body – one or more lines,
terminated by a final line with only a full stop
character>
```

The process examines the message's format after receiving the entire message.
If the format is not correct, then “This is an incorrect format” is printed,
and the three options are given again. If the format is correct, client process
prints the message "Mail sent successfully" on the screen and displays the three
options once again if the mail was sent successfully. If the format is incorrect,
the client give corresponding error information. Keep in mind that mail is kept on
the MailServer machine. The program must interface with the POP3 server running on
the MailServer machine in order to read and delete them.
By establishing a connection with a POP3 server and utilizing the POP3 protocol,
this will be accomplished. You must also determine which POP3 instructions must be
delivered in order to show and remove the message as previously mentioned.

SMTP works on simple text commands/replies that are sent across the TCP connection
between the sender and the receiver. The sender’s commands that we will implement
are the following

`HELO, MAIL_FROM, RCPT_TO, DATA, QUIT.`

And each command has an argument it can accept. The reply codes/error codes
we will implement are the follwing.

```txt
220 <domain name> Service ready
221 <domain> Service closing transmission channel
250 OK <message>
354 Enter mail, end with "." on a line by itself
550 No such user
```

A typical squence of commands in a mail conversation between sender and receiver is
as follows.

```python

client: <client connects to SMTP port>
server: 220 <kuleuven.be> Service ready
client: HELO kuleuven.be
server: 250 OK Hello kuleuven.be
client: MAIL_FROM: <labxyz@kuleuven.be>
server: 250 <labxyz@kuleuven.be>... Sender ok
client: RCPT_TO: otherlab@kuleuven.be
server: 250 root... Recipient ok
client: DATA
server: 354 Enter mail, end with "." on a line by itself
client: From: labxyz@kuleuven.be
client: To: otherlab@kuleuven.be
client: Subject: Test
client: This is a test mail.
client: .
server: 250 OK Message accepted for devivery
client: QUIT
server: 221 kuleuven.be closing connection
client: <client hangs up>

```

## Mail Management

The client program should perform validation on the username and password by
passing `USER` and `PASS` commands to the POP3 server. If authentication is
unsuccessful, an error message should be displayed along with a prompt to re-enter
the credentials. If authentication is successful, a connection should be established
between the client and POP3 server and the client should receive and display a
greeting message from the server: `+OK POP3 server is ready`.

Afterwards, the program receives a list of emails from the POP3 server for the
authenticated user and displays it on the console in the format below:  
`No. <Sender’s email id> <When received, in date: hour : minute> <Subject>`  
The `No.` is the order of the mail in the `my_mailbox` file.
The program provides the following POP3 commands:

-- STAT – count the number of emails.  
-- LIST – should list all the email for the user  
-- RETR – retrieve email based on the serial number  
-- DELE – should delete the email based on the serial number  
-- QUIT – close the connection and terminate the program

Here, the user should provide multiple options in the terminal to enter and accordingly
manage emails. Each of these options will correspond to the commands mentioned above.
When the client sends the QUIT command, the server closes the connection, delivers a
closing message `Thanks for using POP3 server` and cleans up any resources
utilized by the connection.

## Email Message Handling

Email messages are represented using the Email data class, storing attributes
such as sender, recipient, subject, content and size.

```python
@dataclass(frozen=True)
class Email():
    '''
    dataclass used to represent emails.
    number is the email id
    size is the size of the mail in bytes
    (= number of characters since utf8 char = 1 byte)
    '''
    number: int
    sender: str
    to: str
    subject: str
    received: str
    content: str
    size: int

    def to_str(self):
        '''
        Used to convert a mail to str to
        send it to the client.
        '''
        return f"From: {self.sender}\n \
        To: {self.to}\n \
        Subject: {self.subject}\n \
        Received: {self.received}\n \
        {self.content}"

```

The `read_mails` function reads email messages from the user's mailbox,
and the `update_maildrop` function updates the mailbox when a client quits.

```python
def read_mails(username: str, mails: list[Email]):
    '''
    Read the entire mailbox of a user (username) and place the mails
    inside the mails argument.
    '''
    sender = ''
    to = ''
    subject = ''
    received = ''
    content = ''
    mail_size = 0
    with open(f'.\\USERS\\{username}\\my_mailbox.txt') as mailbox:
        for line in mailbox:
            mail_size += len(line) + 1

            # No checks are performed on the order of the mail arguments.
            if line.startswith("From: "):
                sender = line[6:].rstrip()
            elif line.startswith("To: "):
                to = line[4:].rstrip()
            elif line.startswith("Subject: "):
                subject = line[9:].rstrip()
            elif line.startswith("Received: "):
                received = line[10:].rstrip()

            # Line doesn't start with any of the previous, so it is
            # the actual mail content. But if not all previous fields
            # are entered, the mail is of incorrect format.
            elif not (sender and to and subject and received):
                raise Exception("mail format is incorrect.")

            elif line.rstrip() == ".":
                # Much more checks for sender, to, subject, received
                # and content could be performed, but we will assume
                # that the mail was added correctly by the SMTP server.
                if sender and to and subject and received and content:
                    content += line
                    mails.append(Email(len(mails) + 1, sender, to,
                                       subject, received, content, mail_size))
                sender = ''
                to = ''
                subject = ''
                received = ''
                content = ''
                mail_size = 0
            else:
                # Byte-stuffing.
                # Notice that the extra "." doesn't get counted
                # towards the mail size.
                if line.startswith("."):
                    line = "." + line
                content += line

```

```python
def update_maildrop(username: str, mails: list[Email], deleted_mails: list[int]) -> int:
    '''
    update the maildrop: write all mails from the "mails" argument except for the ones who correspond to a deleted mail.
    Returns the resulting size of the maildrop. (can be bigger than what the client worked with since multiple new mails might have arrived,
    and the client doesn't see these when they are looking at their maildrop.)
    '''
    with open(f".\\USERS\\{username}\\my_mailbox.txt", "w") as mailbox:
        for mail in mails:
            if mail.number not in deleted_mails:
                mailbox.write(mail.to_str())
    return len(mails) - len(deleted_mails)

```

## Email message handling

```python
def client_handler(client: so.socket, address: tuple[str, str], global_lock: threading.Lock, file_locks: list[str]):
    '''
    The function that each thread calls on creation. It handles one client connection.
    '''
    # ip and port of connected client.
    ip, port = address
    # client context manager (closes connection if an error would occur and if the context manager is exited.)
    with client:
        thread_variables: ThreadVariables = ThreadVariables(
            client, global_lock, file_locks)
        # set timeout of 10 minutes, if the server does not respond anything for 10 minutes, it will close the connection. (10min. was minimum wait time according to rfc)
        client.settimeout(600)
        # Server welcome message
        client.sendall("+OK POP3 server ready".encode())
        print(f"Client connected from {ip} at port {port}.")
        try:
            while True:
                # recieve client command.
                msg = client.recv(1024).decode()
                # empty message, most likely due to client that closed the connection.
                if not msg:
                    break
                print(f"The client at {ip}:{port} said: '{msg}'")
                # get the command issued by the client.
                match msg.split(" ")[0]:
                    case 'USER':
                        # update thread variables to have current message.
                        thread_variables.msg = msg
                        # handle command
                        USER(thread_variables)
                    case 'PASS':
                        thread_variables.msg = msg
                        PASS(thread_variables)
                    case 'STAT':
                        thread_variables.msg = msg
                        STAT(thread_variables)
                    case 'LIST':
                        thread_variables.msg = msg
                        LIST(thread_variables)
                    case 'RETR':
                        thread_variables.msg = msg
                        RETR(thread_variables)
                    case 'DELE':
                        thread_variables.msg = msg
                        DELE(thread_variables)
                    case 'RSET':
                        thread_variables.msg = msg
                        RSET(thread_variables)
                    case 'QUIT':
                        thread_variables.msg = msg
                        QUIT(thread_variables)
                        break
                    case _:
                        # default case
                        answer = "-ERR unrecognized command."
                        client.sendall(answer.encode())
        except so.timeout:
            print("Client timed out")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            print(
                f"The client from ip: {ip}, and port: {port}, has diconnected!")
            # unlock maildrop (should be unlocked in QUIT(), but if the client pressed CTRL+C instead of quitting first, we will arrive here.)
            thread_variables.global_lock.acquire()
            try:
                thread_variables.locked_files.remove(thread_variables.username)
            # If the maildrop is already unlocked (username already removed from thread_variables.locked_files), a ValueError will be thrown, in this case: do nothing.
            except ValueError:
                pass
            thread_variables.global_lock.release()
```

## Conclusion

The implementation of a Python POP3 server for email management has achieved
its objectives. It provides a functional server that adheres to the POP3 protocol,
allowing users to retrieve and manage email messages.

## Future Work

Future enhancements could include implementing additional POP3 extensions,
improving error handling for edge cases, optimizing server performance further,
and enhancing security features, such as encryption.

## References

- [RFC 1939](https://www.ietf.org/rfc/rfc1939.txt)
- [RFC 821](https://www.ietf.org/rfc/rfc821.txt)
