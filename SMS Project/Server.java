import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;


public class Server {


    private ServerSocket socket;


    public Server(ServerSocket socket){
        this.socket = socket;
    }


    public void openServer() {
        try {
            while (!socket.isClosed()){
                Socket serverSocket = socket.accept();
                System.out.println("Client has entered the chat!");


                ClientHandler CH = new ClientHandler(serverSocket);


                Thread thread = new Thread(CH);
                thread.start();
            }


        } catch (IOException e){


        }
    }


    public void CloseSocket(){
        try{
            if(socket != null){
                socket.close();
            }
        }catch(IOException e){
            e.printStackTrace();
        }
    }


    public static void main(String[] args) throws IOException {


        ServerSocket socket = new ServerSocket(1234);
        Server server = new Server(socket);
        server.openServer();
    }


}
