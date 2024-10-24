import java.io.*;
import java.net.Socket;
import java.util.ArrayList;


public class ClientHandler implements Runnable{


    public static ArrayList<ClientHandler> clientHandlers = new ArrayList<>();
    private Socket socket;
    private BufferedReader bufferedReader;
    private BufferedWriter bufferedWriter;
    private String username;
//    private String Password;


    public ClientHandler(Socket socket){
        try{
            this.socket = socket;
            this.bufferedWriter = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream()));
            this.bufferedReader = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            this.username = bufferedReader.readLine();
//            this.Password = bufferedReader.readLine();
            clientHandlers.add(this);
            broadcastMessage("SERVER: " +username+ " has entered the chat");
        } catch(IOException e) {
            closeAll(socket,bufferedReader,bufferedWriter);
        }
    }


    @Override
    public void run() {
        String messageFromClient;


        while (socket.isConnected()) {
            try{
                messageFromClient = bufferedReader.readLine();
                broadcastMessage(messageFromClient);
            } catch (IOException e){
                closeAll(socket, bufferedReader, bufferedWriter);
                break;
            }
        }


    }


    public void broadcastMessage(String SentMessage) {
        for (ClientHandler clientHandler : clientHandlers) {
            try {
                if(!clientHandler.username.equals(username)) {
                    clientHandler.bufferedWriter.write(SentMessage);
                    clientHandler.bufferedWriter.newLine();
                    clientHandler.bufferedWriter.flush();
                }
            } catch(IOException e) {
                closeAll(socket, bufferedReader, bufferedWriter);
            }
        }
    }


    public void removeClient(){
        clientHandlers.remove(this);
        broadcastMessage("SERVER: "+username+" has left the chat");
    }


    public void closeAll(Socket socket, BufferedReader bufferedReader, BufferedWriter bufferedWriter){
        removeClient();
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
}
