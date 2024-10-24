import javax.swing.*;
import java.io.*;
import java.net.Socket;
import java.util.Scanner;


public class Client {


    private Socket socket;
    private BufferedReader bufferedReader;
    private BufferedWriter bufferedWriter;
    private String username;


    public Client (Socket socket, String username) {
        try{
            this.socket = socket;
            this.bufferedWriter = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream()));
            this.bufferedReader = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            this.username = username;
        } catch (IOException e){
            closeAll(socket, bufferedReader, bufferedWriter);
        }
    }


    public void SendTextMessage() {
        try{
            bufferedWriter.write(username);
            bufferedWriter.newLine();
            bufferedWriter.flush();


            Scanner keyboard = new Scanner(System.in);


            while(socket.isConnected()){
                String SentMessage = keyboard.nextLine();
                if (SentMessage.equalsIgnoreCase("sendfile")){
                    SendFile();
                    bufferedWriter.write("Sent File");

                }
                else {
                    bufferedWriter.write(username + ": " + SentMessage);
                    bufferedWriter.newLine();
                    bufferedWriter.flush();
                }

            }
        }catch (IOException e){
            closeAll(socket, bufferedReader, bufferedWriter);
        }
    }

    public void SendFile() {
        try{
            System.out.println("Please select the file you would like to send!");
//            bufferedWriter.newLine();
//            bufferedWriter.flush();

            JFileChooser j = new JFileChooser();
            j.showOpenDialog(null);

            File filepath = null;
            filepath = j.getSelectedFile();
            System.out.println(filepath);

            FileToServer(filepath);

        } catch(Exception e){}
    }


    public void FileToServer(File filepath) throws IOException {
//        File f = new File(String.valueOf(filepath));
        try{
        if(!filepath.exists()){
            System.out.println("File does not exist!");
        }

        else {
            FileInputStream fileInputStream = new FileInputStream(filepath);
            DataOutputStream outputStream = new DataOutputStream(socket.getOutputStream());

            byte[] buffer = new byte[4096];
            int bytesRead;

            outputStream.writeUTF("FILE:" + filepath.getName());
            outputStream.writeLong(filepath.length());

            while ((bytesRead = fileInputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
//                outputStream.flush();
            }

            File f = new File(/*enter path name*/);
            FileOutputStream fileOutputStream = new FileOutputStream(filepath);
            DataInputStream dataInputStream = new DataInputStream(socket.getInputStream());
            System.out.println(socket.getInputStream());
            long size = dataInputStream.readLong(); // read file size
            int bytes = 0;
            while (size > 0
                    && (bytes = dataInputStream.read(
                    buffer, 0,
                    (int) Math.min(buffer.length, size)))
                    != -1) {
                // Here we write the file using write method
                fileOutputStream.write(buffer, 0, bytes);
                System.out.println(bytesRead);
                size -= bytes; // read upto file size
            }
            System.out.println("File is Received");
            fileOutputStream.close();


            fileInputStream.close();
            newFileCreator(filepath.getAbsolutePath());
        }}catch(Exception e){}
    }

    public void newFileCreator(String filename) throws IOException {
        File ob = new File(filename + ".txt");
        boolean filecreated = false;
        try{
            filecreated = ob.createNewFile();
        }
        catch(IOException e){
            System.out.println("Error" + e);
        }

        if(filecreated == true){
            System.out.println("Created file" + ob.getPath());




        }
    }

    public void getMessages() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                String GroupMessages;


                while(socket.isConnected())
                {
                    try {
                        GroupMessages = bufferedReader.readLine();
                        System.out.println(GroupMessages);
                    }catch(IOException e){
                        closeAll(socket, bufferedReader, bufferedWriter);
                    }
                }


            }
        }).start();
    }


    public void closeAll(Socket socket, BufferedReader bufferedReader, BufferedWriter bufferedWriter){
        try {
            if (bufferedReader != null) {
                bufferedReader.close();
            }
            if (bufferedWriter != null) {
                bufferedWriter.close();
            }
            if (socket != null) {
                socket.close();
            }
        } catch(IOException e){
            e.printStackTrace();
        }
    }


    public static void main(String[] args) throws IOException {


        Scanner keyboard = new Scanner(System.in);
        System.out.println("Please enter your username: ");
        String username = keyboard.nextLine();
        Socket socket = new Socket("localhost", 1234);
        Client client = new Client(socket, username);
        client.getMessages();
        client.SendTextMessage();




    }
}
