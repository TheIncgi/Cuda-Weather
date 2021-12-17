package app.util;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;

import javax.imageio.ImageIO;

public class FontSpriteGenerator {
	//Epilogue
	public static void main(String[] args) throws NumberFormatException, IOException {
		if(args.length > 0) {
			makeFontSprite(args[0], args.length >= 2 ? Integer.parseInt(args[1]) : 32);
		}else{
			makeFontSprite("epilogue", 24);
		}
	}
	
	public static void makeFontSprite(String fontName, int pixelHeight) throws IOException {
		BufferedImage img = new BufferedImage(pixelHeight, pixelHeight, BufferedImage.TYPE_INT_ARGB);
		var g = img.createGraphics();
		g.setFont(new Font(fontName, 0, 20));
		var m = g.getFontMetrics();
		float factor = ((float)pixelHeight) / (m.getAscent() + m.getDescent());
		g.setFont(new Font(fontName, 0, (int)(20 * factor)));
		
		StringBuilder all = new StringBuilder();
		for(int i = 32; i<=127; i++) {
			all.append((char)i);
		}
		
		int twid = 0;
		for(int i = 32; i<=127; i++) {twid += g.getFontMetrics().charWidth((char)i);}
		img = new BufferedImage(twid, pixelHeight, BufferedImage.TYPE_INT_ARGB);
		g = img.createGraphics();
		g.setFont(new Font(fontName, 0, (int)(20 * factor)));
		g.setColor(Color.BLACK);
		
		int[] starts = new int[127-32+1+1];
		int w = 0;
		int baseline = g.getFontMetrics().getAscent();
		for(int i = 32; i<=127; i++) {
			starts[i-32] = w;
			char c = (char) i;
			g.drawString(c+"", w+1, baseline);
			w += g.getFontMetrics().charWidth(c);
		}
		starts[127-32] = w;
		
		ImageIO.write(img, "png", new File("sprite.png"));
		
		RandomAccessFile raf = new RandomAccessFile(fontName+"_"+pixelHeight+".cuFont","rw");
		raf.writeInt(img.getWidth());
		raf.writeInt(img.getHeight());
		for (int i = 0; i < starts.length; i++) {
			raf.writeInt(starts[i]);
		}
		
		for(int y=0; y<img.getHeight(); y++) {
			for(int x=0; x<img.getWidth(); x++) {
				raf.writeByte(img.getRGB(x, y) == 0xFF000000 ? 1 : 0);
			}
		}
		raf.close();
	}

	
}
